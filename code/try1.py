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



    X1_Vec = positive
    X2_Vec = negative


    y1 = np.zeros(len(positive), dtype=int)
    y2 = np.ones(len(negative), dtype=int)

    return X1_Vec,X2_Vec, y1, y2
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

def recall_m(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# 读取原始数据集，分词。
pchinese = re.compile('([\u4e00-\u9fa5]+)+?')
X1_Vec, X2_Vec, y1, y2 = loadfile()



def train_test1(X_Vec, y):


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
    x_train, x_val, y_train, y_val = train_test_split(index, y, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(index, y, test_size=2/3)
    # 设定一些参数。
    max_words = 20000
    maxlen = 100
    embed_dim = 1000
    batch_size = 64

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
    model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy', f1_score])



    # 转换一下输入数据的类型。
    x_train = np.asarray(x_train).astype('float64')
    y_train = np.asarray(y_train).astype('float64')
    x_val = np.asarray(x_val).astype('float64')
    y_val = np.asarray(y_val).astype('float64')

    # 训练。
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=20, validation_data=(x_val, y_val))


    # 测试
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    x_test = np.asarray(x_test).astype('float64')
    y_test = np.asarray(y_test).astype('float64')
    test_loss, test_acc, test_f1 = model.evaluate(x_test, y_test)

train_test1(X2_Vec, y2)

# 拿一句话来测试看看。
# in_str = "空气中弥漫着一股浓浓放假的味道憨笑憨笑憨笑都准备去哪里哈皮哇跳跳转圈"
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
# data=sequence.pad_sequences(data, maxlen=maxlen )
# pre=model.predict(data)[0].tolist()
# # print(pre)
# print("输入：")
# print("  ",in_str)
# print("        ")
# print("输出:")
# label={0:"positive",1:"negative"}
# print("  ",label[pre.index(max(pre))])
# 目前不知道为什么它只会输出positive。后面暂时不管了.