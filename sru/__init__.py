from .version import __version__
import torch
import torch.nn as nn
from torch.autograd import Function, Variable

use_gpu = True
try:
    from ._ext import sru_cu
except ImportError as e:
    print('cuda sru error %s, running in CPU mode.' % e)
    use_gpu = False
    sru_cu = None


class SRUComputeGPU(Function):

    def __init__(self, activation_type, d_out, bidirectional=False):
        super(SRUComputeGPU, self).__init__()
        self.activation_type = activation_type
        self.d_out = d_out
        self.bidirectional = bidirectional
        self.intermediate = None

    def forward(self, u, x, bias, init=None, mask_h=None):
        directions = 2 if self.bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        n_cols = batch * d * directions

        init_ = x.new(n_cols).zero_() if init is None else init
        size = (length, batch, d * directions) if x.dim() == 3 else (batch, d * directions)
        c = x.new(*size)
        h = x.new(*size)

        func = sru_cu.sru_forward_cuda if not self.bidirectional else sru_cu.sru_bi_forward_cuda

        func(u.contiguous(), x.contiguous() if k_ == 3 else torch.Tensor([0]).cuda(), bias,
             init_.contiguous(), mask_h if mask_h is not None else torch.Tensor([0]).cuda(), h, c,
             length, batch, d, k_, self.activation_type)

        self.save_for_backward(u, x, bias, init, mask_h)
        self.intermediate = c
        if x.dim() == 2:
            last_hidden = c
        elif self.bidirectional:
            last_hidden = torch.cat((c[-1, :, :d], c[0, :, d:]), dim=1)
        else:
            last_hidden = c[-1]

        return h, last_hidden

    def backward(self, grad_h, grad_last):
        directions = 2 if self.bidirectional else 1
        u, x, bias, init, mask_h = self.saved_tensors
        c = self.intermediate
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        n_cols = batch * d * directions

        init_ = x.new(n_cols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_bias = x.new(2, batch, d * directions)
        grad_init = x.new(batch, d * directions)

        # For DEBUG
        # size = (length, batch, x.size(-1)) if x.dim() == 3 else (batch, x.size(-1))
        # grad_x = x.new(*x.size()) if k_ == 3 else x.new(*size).zero_()

        # Normal use
        grad_x = x.new(*x.size()) if k_ == 3 else None

        func = sru_cu.sru_backward_cuda if not self.bidirectional else sru_cu.sru_bi_backward_cuda
        func(u.contiguous(),
             x.contiguous() if k_ == 3 else torch.Tensor([0]).cuda(),
             bias, init_.contiguous(),
             mask_h if mask_h is not None else torch.Tensor([0]).cuda(),
             c, grad_h.contiguous(), grad_last.contiguous(),
             grad_bias, grad_init, grad_u, grad_x if k_ == 3 else torch.Tensor([0]).cuda(),
             length, batch, d, k_, self.activation_type)
        return grad_u, grad_x, grad_bias.sum(1).view(-1), grad_init, None


def sru_cpu_compute(activation_type, d, bidirectional=False):
    """CPU version of the core SRU computation.

    Has the same interface as SRU_Compute_GPU() but is a regular Python function
    instead of a torch.autograd.Function because we don't implement backward()
    explicitly.
    """

    def sru_compute_cpu(u, x, bias, init=None, mask_h=None):
        directions = 2 if bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        k = u.size(-1) // d // directions

        if mask_h is None:
            mask_h = 1

        u = u.view(length, batch, directions, d, k)

        x_tilde = u[..., 0]

        forget_bias, reset_bias = bias.view(2, directions, d)
        forget = (u[..., 1] + forget_bias).sigmoid()
        reset = (u[..., 2] + reset_bias).sigmoid()

        if k == 3:
            x_prime = x.view(length, batch, directions, d)
        else:
            x_prime = u[..., 3]

        h = Variable(x.data.new(length, batch, directions, d))

        if init is None:
            c_init = Variable(x.data.new(batch, directions, d).zero_())
        else:
            c_init = init.view(batch, directions, d)

        c_final = []
        for di in range(directions):
            if di == 0:
                time_seq = range(length)
            else:
                time_seq = range(length - 1, -1, -1)

            c_prev = c_init[:, di, :]
            c_t = None
            for t in time_seq:
                c_t = (c_prev - x_tilde[t, :, di, :]) * forget[t, :, di, :] + x_tilde[t, :, di, :]
                c_prev = c_t

                if activation_type == 0:
                    g_c_t = c_t
                elif activation_type == 1:
                    g_c_t = c_t.tanh()
                elif activation_type == 2:
                    g_c_t = nn.functional.relu(c_t)
                else:
                    assert False, 'Activation type must be 0, 1, or 2, not {}'.format(activation_type)

                h[t, :, di, :] = (g_c_t * mask_h - x_prime[t, :, di, :]) * reset[t, :, di, :] + x_prime[t, :, di, :]

            c_final.append(c_t)

        return h.view(length, batch, -1), torch.stack(c_final, dim=1).view(batch, -1)

    return sru_compute_cpu


class SRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, rnn_dropout=None, activation_type=1):
        super(SRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout if rnn_dropout else dropout
        self.activation_type = activation_type
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self.num_directions = num_directions
        output_size = hidden_size * num_directions
        k = 4 if input_size != output_size else 3
        self.hidden_size_per_direction = hidden_size * k
        self.weight = nn.Parameter(torch.Tensor(input_size, self.hidden_size_per_direction * num_directions))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * num_directions * 2))
        self.init_weight()

    def init_weight(self):
        val_range = (3.0 / self.input_size) ** 0.5
        self.weight.data.uniform_(-val_range, val_range)
        self.bias.data.zero_()

    def set_bias(self, bias_val=0):
        n_out = self.hidden_size
        self.bias.data[n_out * self.num_directions:].zero_().add_(bias_val)

    def forward(self, input, c0=None):
        assert input.dim() == 2 or input.dim() == 3
        input_size, output_size = self.input_size, self.hidden_size
        batch = input.size(-2)
        if c0 is None:
            c0 = Variable(input.data.new(batch, output_size * self.num_directions).zero_())

        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch, input_size), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, input_size)
        u = x_2d.mm(self.weight)

        if input.is_cuda and use_gpu:
            sru_compute = SRUComputeGPU(self.activation_type, output_size, self.bidirectional)
        else:
            sru_compute = sru_cpu_compute(self.activation_type, output_size, self.bidirectional)

        if self.training and (self.dropout > 0):
            mask_h = self.get_dropout_mask_((batch, output_size * self.num_directions), self.dropout)
            return sru_compute(u, input, self.bias, c0, mask_h)
        else:
            return sru_compute(u, input, self.bias, c0)

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1 - p).div_(1 - p))


class SRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0, rnn_dropout=0,
                 bidirectional=False, activation_type=1):
        super(SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.out_size = hidden_size * 2 if bidirectional else hidden_size
        num_directions = 2 if bidirectional else 1

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * num_directions
            cell = SRUCell(layer_input_size, hidden_size, dropout=dropout if layer + 1 != num_layers else 0,
                           rnn_dropout=rnn_dropout, bidirectional=bidirectional, activation_type=activation_type)
            self.rnn_lst.append(cell)

    def set_bias(self, bias_val=0):
        for l in self.rnn_lst:
            l.set_bias(bias_val)

    def forward(self, input, c0=None, return_hidden=True):
        assert input.dim() == 3  # (len, batch, n_in)
        num_directions = 2 if self.bidirectional else 1
        if c0 is None:
            zeros = Variable(input.data.new(input.size(1), self.hidden_size * num_directions).zero_())
            c0 = [zeros for _ in range(self.depth)]
        else:
            assert c0.dim() == 3  # (depth, batch, n_out*dir_)
            c0 = [x.squeeze(0) for x in c0.chunk(self.depth, 0)]

        prev_x = input
        last_c = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prev_x, c0[i])
            prev_x = h
            last_c.append(c)

        if return_hidden:
            return prev_x, torch.stack(last_c)
        else:
            return prev_x

