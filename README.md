# 基于transformer的情感分类器
## 步骤

### 1、运行train_test2.py

即可显示运行结果，包括类别准确率以及F1值。


## FAQ
###1、如何修改embed_dim？

将train_test2.py中第135行的embed_dim，以及word2vec.py的voc_dim，修改为想要的单词维度即可。


### 2、如何进行标签为positive的文本的训练、预测？


将train_test2.py中第175行的X2_Vec, y2改为X1_Vec, y1，即可。

### 3、train_f1是做什么的？
这个是没有分开positive和negative的训练、测试脚本，也可以成功运行。

### 4、环境呢？
需要什么就pip什么，版本不对的话就谷歌一下看看正确的版本是什么，然后再pip一下就好了。

后面记得加上

-i https://pypi.tuna.tsinghua.edu.cn/simple

这样快一点。

