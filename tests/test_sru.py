import copy
import numpy as np
import torch
from torch.autograd import Variable
from sru import SRU as CuSRU
from sru.cuda_functional import SRU

np.set_printoptions(suppress=True)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

x0 = np.random.random_sample((2, 4, 6))
x = Variable(torch.FloatTensor(x0))

input_size, hidden_size = 6, 8

rnn1 = CuSRU(input_size, hidden_size, num_layers=2, bidirectional=True)
states = rnn1.state_dict()
o1, c1 = rnn1(x)

rnn2 = CuSRU(input_size, hidden_size, num_layers=2, bidirectional=True)
rnn2.load_state_dict(states)
if torch.cuda.is_available():
    rnn2.cuda()
    x2 = x.cuda()
else:
    x2 = x
o2, c2 = rnn2(x2)

rnn3 = SRU(input_size, hidden_size, num_layers=2, bidirectional=True)
rnn3.load_state_dict(states)
o3, c3 = rnn3(x)

rnn4 = SRU(input_size, hidden_size, num_layers=2, bidirectional=True)
rnn4.load_state_dict(states)
if torch.cuda.is_available():
    rnn4.cuda()
    x4 = x.cuda()
else:
    x4 = x
o4, c4 = rnn4(x4)

po1 = np.allclose(o1.data.cpu().numpy(), o2.data.cpu().numpy())
pc1 = np.allclose(c1.data.cpu().numpy(), c2.data.cpu().numpy())

print('cu_sru_cpu == cu_sru_gpu: %s' % (po1 and pc1))

po2 = np.allclose(o1.data.cpu().numpy(), o3.data.cpu().numpy())
pc2 = np.allclose(c1.data.cpu().numpy(), c3.data.cpu().numpy())

print('cu_sru_cpu == cuda_functional_cpu: %s' % (po2 and pc2))

po3 = np.allclose(o1.data.cpu().numpy(), o4.data.cpu().numpy())
pc3 = np.allclose(c1.data.cpu().numpy(), c4.data.cpu().numpy())

print('cu_sru_cpu == cuda_functional_gpu: %s' % (po3 and pc1))
