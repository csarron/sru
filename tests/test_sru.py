import copy
import numpy as np
import torch
from torch.autograd import Variable
from sru import SRU as CuSRU
from cuda_functional import SRU

np.set_printoptions(suppress=True)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

x0 = np.random.random_sample((2, 4, 6))
x = Variable(torch.FloatTensor(x0))

input_size, hidden_size = 6, 8

rnn1 = CuSRU(input_size, hidden_size, num_layers=2, bidirectional=True)
params = {'states': rnn1.state_dict()}
torch.save(params, '/tmp/m.pt')
o1, c1 = rnn1(x)

saved_params = torch.load('/tmp/m.pt', map_location=lambda storage, loc: storage)

rnn2 = CuSRU(input_size, hidden_size, num_layers=2, bidirectional=True)
rnn2.cuda()
rnn2.load_state_dict(saved_params['states'])
o2, c2 = rnn2(x.cuda())

rnn3 = SRU(input_size, hidden_size, num_layers=2, bidirectional=True)
rnn3.load_state_dict(saved_params['states'])
o3, c3 = rnn3(x)

rnn4 = SRU(input_size, hidden_size, num_layers=2, bidirectional=True)
rnn4.cuda()
rnn4.load_state_dict(saved_params['states'])
o4, c4 = rnn4(x.cuda())

print(np.allclose(o1.data.cpu().numpy().flatten(), o2.data.cpu().numpy().flatten()))
print(np.allclose(c1.data.cpu().numpy().flatten(), c2.data.cpu().numpy().flatten()))

print(np.allclose(o3.data.cpu().numpy().flatten(), o4.data.cpu().numpy().flatten()))
print(np.allclose(c3.data.cpu().numpy().flatten(), c4.data.cpu().numpy().flatten()))

print(np.allclose(o1.data.cpu().numpy().flatten(), o3.data.cpu().numpy().flatten()))
print(np.allclose(c1.data.cpu().numpy().flatten(), c3.data.cpu().numpy().flatten()))

print(np.allclose(o2.data.cpu().numpy().flatten(), o4.data.cpu().numpy().flatten()))
print(np.allclose(c2.data.cpu().numpy().flatten(), c4.data.cpu().numpy().flatten()))
# print('cuda_functional_cpu:%s, cuda_functional_gpu:%s, cu_sru_cpu:%s, cu_sru_gpu:%s' % ())
