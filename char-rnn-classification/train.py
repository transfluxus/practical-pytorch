# encoding: utf-8 
import torch
from data import *
from model import *
import random
import time
import math

n_hidden = 128
n_epochs = 100000
print_every = 5000
plot_every = 1000
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

import argparse


parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

print("USING CUDA" if args.cuda else "NO CUDA")

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    cat_tensor = torch.LongTensor([all_categories.index(category)])
    line_tensor = lineToTensor(line)
    if args.cuda:
        cat_tensor = cat_tensor.pin_memory().cuda()
        line_tensor = line_tensor.pin_memory().cuda()

    category_tensor_var = Variable(cat_tensor)
    line_tensor_var = Variable(line_tensor)
    return category, line, category_tensor_var, line_tensor_var

rnn = RNN(n_letters, n_hidden, n_categories)
if args.cuda:
    rnn.cuda()

optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden(args.cuda)
    # if args.cuda:
    #     hidden = hidden.cuda()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        # print(line_tensor[i], hidden)
        output, hidden = rnn(line_tensor[i], hidden)
        # if args.cuda:
        #     hidden = hidden.cuda()

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data[0]

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = randomTrainingPair()
    # if args.cuda:
    #     category_tensor = category_tensor.cuda()
    #     line_tensor = line_tensor.cuda()

    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '+++' if guess == category else ('--- / ' + category)
        #print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))
        print(epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct)
    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn, 'char-rnn-classification.pt')

