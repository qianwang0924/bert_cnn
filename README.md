# bert_cnn
上传了数据集，还在制作中
2020/3/9
数据集加上数据清洗制作完成，数据集有三个人不同的情感标签评论，对于三个人情感标签完全不相同的情况，将其剔除，不纳入训练的数据集中。
2020/3/10
实现了cnn部分，为简化参数量，将VGGnet最后的全连接层实现全局池化。下一步的工作是探索怎么讲cnn与bert联系起来。用训练标签一个损失函数来实现两个网络的同时优化。
？或者先分别单独训练一个，然后两个和一起拿来微调，主要是数据集太复杂了？
