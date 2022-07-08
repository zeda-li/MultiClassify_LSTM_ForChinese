import re
import jieba
import numpy as np
from word2vec import word2vec_train
from lstm import lstm
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.preprocessing import sequence
import yaml

from keras import backend as K
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from tensorflow.keras.optimizers import Adam
from keras import Model

from PositionEmbedding import SinusoidalPositionEmbedding
from MultiHeadAttention import MultiHeadAttention
from LayerNormalization import LayerNormalization
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
def transformer_encoder(inputs, num_heads=4, dropout_rate=0.1):
    in_dim = K.int_shape(inputs)[-1]
    x = MultiHeadAttention(num_heads, in_dim)([inputs, inputs])
    x = Dropout(dropout_rate)(x)
    x = add([inputs, x])
    x1 = LayerNormalization()(x)
    x = Dense(in_dim * 2, activation='relu')(x1)
    x = Dense(in_dim)(x)
    x = Dropout(dropout_rate)(x)
    x = add([x1, x])
    x = LayerNormalization()(x)
    return x
def data2inx(w2indx,X_Vec):
    data = []
    for sentence in X_Vec:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)

        data.append(new_txt)
    return data

# 读取原始数据集，分词。
pchinese = re.compile('([\u4e00-\u9fa5]+)+?')
X_Vec,y=loadfile()
input_dim, embedding_weights, w2dic = word2vec_train(X_Vec)
# X_Vec里面就是每一句话是一个列表，然后组成的大列表。注意这里的每一句话都是已经经过分词，但是仍然是中文的。X_Vec已经装载了所有的训练集的内容，1032之前是positive的，1032之后是negative的。
# y是与X_Vec里面的句子对应的很好的。
# input_dim是词表里面有多少个词，就是那个1100，
# embedding_weights是1100*150的，就是那个里面一堆数字的词表，150是每个词用150维的向量表示。
# w2dic是具体中文词和具体中文词对应的词号组成的词表。

# 划分为训练集、测试集。
index = data2inx(w2dic,X_Vec)
# index里面就是每一句话就是一个列表，然后组成的大列表。注意这里的每一句话是由词号构成的列表。



# 划分训练集和验证集。
x_train, x_val, y_train, y_val = train_test_split(index, y, test_size=0.2)

# 设定一些参数。
max_words = 20000
maxlen = 100
embed_dim = 64
batch_size = 128

# 让每个句子的长度相同。
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
# 不然会报ValueError: setting an array element with a sequence.的错。


# 定义网络
text_input = Input(shape=(maxlen,), dtype='int32')
x = Embedding(max_words, embed_dim)(text_input)
x = SinusoidalPositionEmbedding()(x)
x = transformer_encoder(x)
x = GlobalAveragePooling1D()(x)
out = Dense(1, activation='sigmoid')(x)

# 定义模型
model = Model(text_input, out)
model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 转换一下输入数据的类型。
x_train = np.asarray(x_train).astype('float64')
y_train = np.asarray(y_train).astype('float64')
x_val = np.asarray(x_val).astype('float64')
y_val = np.asarray(y_val).astype('float64')

# 训练。
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=10, validation_data=(x_val, y_val))



# 拿一句话来测试看看。
# in_str = "实在无法忍受了，我们大打出手，他嚣张的气焰令我无比的不爽"
# in_stc = ''.join(pchinese.findall(in_str))
# in_stc = list(jieba.cut(in_stc, cut_all=True, HMM=False))
# new_txt = []
# data = []
# for word in in_stc:
#     try:
#         new_txt.append(w2dic[word])
#     except:
#         new_txt.append(0)
# data.append(new_txt)
# data=sequence.pad_sequences(data, maxlen=voc_dim )
# pre=model.predict(data)[0].tolist()
# print(pre)
# print("输入：")
# print("  ",in_str)
# print("        ")
# print("输出:")
# print("  ",label[pre.index(max(pre))])
