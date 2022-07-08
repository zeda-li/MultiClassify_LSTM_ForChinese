import re
import jieba
import numpy as np
from word2vec import word2vec_train
from lstm import lstm
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.preprocessing import sequence
import yaml

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
clean_data('../data/positive.txt','../data/positive_clean.txt')
clean_data('../data/negative.txt','../data/negative_clean.txt')
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

pchinese = re.compile('([\u4e00-\u9fa5]+)+?')

if __name__=="__main__":
    X_Vec,y=loadfile()
    print(y)
    input_dim, embedding_weights, w2dic = word2vec_train(X_Vec)
    print(input_dim,embedding_weights,w2dic)

    in_str = "实在无法忍受了，我们大打出手，他嚣张的气焰令我无比的不爽"
    in_stc = ''.join(pchinese.findall(in_str))
    in_stc = list(jieba.cut(in_stc, cut_all=True, HMM=False))
    new_txt = []
    data = []
    for word in in_stc:
        try:
            new_txt.append(w2dic[word])
        except:
            new_txt.append(0)
    for j in new_txt:
        print(len(embedding_weights[j]))
        print(embedding_weights[j])