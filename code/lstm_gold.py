import os
import torch
import csv
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as f

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
reader = csv.reader(open("LBMA-GOLD.csv"))
data = []
for time, value in reader:
    if value == "":
        continue
    if time == "Date":
        continue
    data.append(float(value))
scale = max(data)
data = list(map(lambda x: x / max(data), data))


class LSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.wi = torch.nn.Parameter(torch.randn([hidden_size, input_size + hidden_size]))
        self.bi = torch.nn.Parameter(torch.randn([hidden_size, input_size]))
        self.wf = torch.nn.Parameter(torch.randn([hidden_size, input_size + hidden_size]))
        self.bf = torch.nn.Parameter(torch.randn([hidden_size, input_size]))
        self.wo = torch.nn.Parameter(torch.randn([hidden_size, input_size + hidden_size]))
        self.bo = torch.nn.Parameter(torch.randn([hidden_size, input_size]))
        self.wc = torch.nn.Parameter(torch.randn([hidden_size, input_size + hidden_size]))
        self.bc = torch.nn.Parameter(torch.randn([hidden_size, input_size]))
        self.out = torch.nn.Parameter(torch.randn([output_size, hidden_size]))
        self.sig = f.sigmoid
        self.tanh = f.tanh

    def forward(self, in_put, hidden, ct_last):
        combined = torch.cat((in_put, hidden), 0)
        it = self.sig(self.wi.matmul(combined) + self.bi)
        ft = self.sig(self.wf.matmul(combined) + self.bf)
        ot = self.sig(self.wo.matmul(combined) + self.bo)
        ct_ = self.tanh(self.wc.matmul(combined) + self.bc)
        ct = ft * ct_last + it * ct_
        ht = ot * self.tanh(ct)
        output = self.out.matmul(ht)
        return output, ht, ct

    def initHidden(self):
        return torch.zeros(self.hidden_size, 1)

    def initct(self):
        return torch.zeros(self.output_size, 1)


def train():
    if os.path.exists("lstm_parameters"):
        lstm = torch.load("lstm_parameters")
    else:
        lstm = LSTM(1, 1, 7)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
    x_tensor = torch.tensor(data)
    for epoch in range(2000):
        output = torch.zeros(len(data) + 1)
        output[0] = torch.tensor(data[0])
        hidden = lstm.initHidden()
        ct = lstm.initct()
        for i in range(len(data)):
            input = x_tensor[i].resize(1, 1)
            out, hidden, ct = lstm(input, hidden, ct)
            output[i + 1] = out.squeeze()
        print(output)
        optimizer.zero_grad()
        loss = criterion(output[:-1:], x_tensor)
        loss.backward(retain_graph=True)
        optimizer.step()
        if epoch > 1 and epoch % 10 == 0:
            print("第{}次训练结束".format(epoch))
            torch.save(lstm, "lstm_parameters")

def test():
    if os.path.exists("lstm_parameters"):
        lstm = torch.load("lstm_parameters")
    else:
        return "Error"
    output = torch.zeros(len(data) + 1)
    output[0] = torch.tensor(data[0])
    x_tensor = torch.tensor(data)
    lth = int(len(data)*0.8)
    hidden = lstm.initHidden()
    ct = lstm.initct()
    for i in range(len(data)):
        input = x_tensor[i].resize(1, 1)
        out, hidden, ct = lstm(input, hidden, ct)
        output[i + 1] = out.squeeze()
    x = output.detach().numpy()
    a = (x * scale)[lth::]
    b = (np.array(data) * scale)[lth::]
    # a = (x * scale)
    # b = (np.array(data) * scale)
    plt.plot(a, color='red', label='predict')
    plt.plot(b, color='blue', label='truth')
    plt.xlabel("Days")
    plt.ylabel("Dollars")
    plt.legend()
    plt.show()

test()