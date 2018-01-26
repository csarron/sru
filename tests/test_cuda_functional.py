import numpy as np
import torch
from torch.autograd import Variable
from sru import SRU

np.set_printoptions(suppress=True)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

x0 = np.random.random_sample((2, 4, 6))
x = Variable(torch.FloatTensor(x0))

input_size, hidden_size = 6, 8

rnn = SRU(input_size, hidden_size, num_layers=2, bidirectional=True)
states = rnn.state_dict()
o, c = rnn(x)

rnn2 = SRU(input_size, hidden_size, num_layers=2, bidirectional=True)
rnn2.cuda()
rnn2.load_state_dict(states)
o2, c2 = rnn2(x.cuda())

print(o)
print(o2)
print(np.allclose(o.data.cpu().numpy().flatten(), o2.data.cpu().numpy().flatten()))
print(np.allclose(c.data.cpu().numpy().flatten(), c2.data.cpu().numpy().flatten()))
print('cuda_functional gpu')
