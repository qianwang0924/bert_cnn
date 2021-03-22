# bert_cnn
上传了数据集，还在制作中
2020/3/9
数据集加上数据清洗制作完成，数据集有三个人不同的情感标签评论，对于三个人情感标签完全不相同的情况，将其剔除，不纳入训练的数据集中。
2020/3/10
实现了cnn部分，为简化参数量，将VGGnet最后的全连接层实现全局池化。下一步的工作是探索怎么讲cnn与bert联系起来。用训练标签一个损失函数来实现两个网络的同时优化。
？或者先分别单独训练一个，然后两个和一起拿来微调，主要是数据集太复杂了？
2020/3/22
完成了代码的调试，发现bert输出的向量如果融合cnn，在网络中cnn中代码初始化会殃及bert。bert的权重是预训练好的，打算用迁移学习的方法来实现训练。所以不能初始化。现在就是找到一种不能让bert初始化的方法。
 Error while reading resource variable tf_bert_model/bert/encoder/layer_._5/attention/self/value/kernel from Container: localhost. This could mean that the variable was uninitialized. Not found: Container localhost does not exist. (Could not find resource: localhost/tf_bert_model/bert/encoder/layer_._5/attention/self/value/kernel)
