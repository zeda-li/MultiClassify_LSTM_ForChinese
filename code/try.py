import re
import jieba
import numpy as np
from word2vec import word2vec_train
from lstm import lstm
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.preprocessing import sequence
import yaml
import os
import torch
import csv
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as f
def clean_data(rpath,wpath):
    # coding=utf-8
    pchinese = re.compile('([\u4e00-\u9fa5]+)+?')
    f = open(rpath,encoding="UTF-8")
    fw = open(wpath, "w",encoding="UTF-8")
    for line in f.readlines():
        m = pchinese.findall(str(line))
        if m:
            str1 = ''.join(m)
            str2 = str(str1)
            fw.write(str2)
            fw.write("\n")
    f.close()
    fw.close()
# clean_data('../data/positive.txt','../data/positive_clean.txt')
# clean_data('../data/negative.txt','../data/negative_clean.txt')
def loadfile():
    positive=[]
    negative=[]
    with open('../data/positive_clean.txt',encoding='UTF-8') as f:
        for line in f.readlines():
            positive.append(list(jieba.cut(line, cut_all=False, HMM=True))[:-1])
        f.close()
    with open('../data/negative_clean.txt',encoding='UTF-8') as f:
        for line in f.readlines():
            negative.append(list(jieba.cut(line, cut_all=False, HMM=True))[:-1])

    X_Vec = np.concatenate((positive,negative))

    y = np.concatenate((np.zeros(len(positive), dtype=int),
                        np.ones(len(negative), dtype=int)))

    return X_Vec, y

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
        return torch.zeros(self.hidden_size)

    def initct(self):
        return torch.zeros(self.output_size)
def train(model, epoches, data):
    lstm = model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
    x_tensor = torch.tensor(data)
    for epoch in range(epoches):
        hidden = lstm.initHidden()
        ct = lstm.initct()
        for i in range(len(data)):
            input = x_tensor[i][0]
            for j in range(len(data[i])):
                out, hidden, ct = lstm(input, hidden, ct)

        print(out)
        optimizer.zero_grad()
        loss = criterion(out, x_tensor)
        loss.backward()
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

if __name__=="__main__":

    # 读取给的两个数据集。
    pchinese = re.compile('([\u4e00-\u9fa5]+)+?')
    X_Vec,y=loadfile()
    print(len(X_Vec[0]))

    input_dim, embedding_weights, w2dic = word2vec_train(X_Vec)
    print(input_dim,embedding_weights,w2dic)


    # 测试一句话并且显示它里面每个词的词号组成的列表看一看。
    in_str = "实在无法忍受了，我们大打出手，他嚣张的气焰令我无比的不爽"
    in_stc = ''.join(pchinese.findall(in_str))
    in_stc = list(jieba.cut(in_stc, cut_all=True, HMM=False))
    new_txt = []
    for word in in_stc:
        try:
            new_txt.append(w2dic[word])
        except:
            new_txt.append(0)
    print(new_txt)





    # 训练
    data = torch.ones([500, 8, 150])
    print(len(data), len(data[0]))
    model = LSTM(len(data[0]), 2, 256)
    epoches=20
    train(model, epoches, data)
