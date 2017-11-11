import torch
import torch.nn as nn
from torch.autograd import Variable

# from train import args 

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        # hidden = hidden.cuda()
        combined = torch.cat((input, hidden), 1)
        # print(combined)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        # print('DONE')
        return output, hidden

    def initHidden(self, cuda):
        hidden_tensor = torch.zeros(1, self.hidden_size)
        if cuda:
            hidden_tensor = hidden_tensor.pin_memory().cuda()
        return Variable(hidden_tensor)
