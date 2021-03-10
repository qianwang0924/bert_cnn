import tensorflow as tf
from transformers import BertTokenizer #该类实现数据的token化
from transformers import TFBertForSequenceClassification
import os
import pandas as pd
import numpy as np
from PIL import Image


max_sentence_length = 40 #定义句子最大的长度
label_path = 'D:\code\\bert\word_data2\labelResultAll.txt'
data_dir_path = 'D:\code\\bert\word_data2\data'
number_of_epochs = 8 #训练循环次数
learning_rate = 2e-5 #学习率


def get_bert_input_V1(test_sentence,max_sentence_length):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')#加载bert实现序列化预训练模型

    test_sentence_with_special_tokens = '[CLS]' + test_sentence + '[SEP]'
    tokenized = tokenizer.tokenize(test_sentence_with_special_tokens)
    print('tokenized', tokenized)

    input_ids = tokenizer.convert_tokens_to_ids(tokenized)#实现token转化为id
    #print('tokenized_id',input_ids)

    #实现填充句子最大长度
    padding_length = max_sentence_length - len(input_ids)
    for i in range(padding_length):
        input_ids.append(0)
        
    #实现句子的
    #print('input_ids',input_ids)
    attention_mask=[]
    for i in tokenized:
        attention_mask.append(1)
    for j in range(padding_length):
        attention_mask.append(0)
    #print('attention_mask',attention_mask)

    #实现分别token的种类，为后面bert模型输入做好准备
    token_type_ids = [0] *max_sentence_length
    #print('token_type_ids',token_type_ids)
    bert_input = {
        "token_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask
    } 
    print(bert_input)
    return(bert_input)

def get_bert_input_V2(sentence,max_sentence_length):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')#加载bert实现序列化预训练模型
    bert_input = tokenizer.encode_plus(sentence,add_special_tokens = True,#增加句子的前后缀
                            max_length = max_sentence_length,#定义句子的最大长度
                            pad_to_max_length = True,#是否填充到一致长度
                            return_attention_mask = True)#实现attention_mask
    #print(bert_input)
    return bert_input



###########################################数据集处理模块################################################
def get_label(label_path):
    pre_train_label = pd.read_csv(label_path,sep="\t",header=None,encoding="gbk")
    df = pd.DataFrame(pre_train_label)
    text_label = []
    image_label = []

    flag = []
    id_use = []
    id_unuse = []
    id_idx_unuse = []

    #遍历整个行，一共有19600条数据
    for i in range(1,19600):
        if(pre_train_label[1][i] == pre_train_label[2][i] or pre_train_label[1][i] == pre_train_label[3][i] or pre_train_label[2][i] == pre_train_label[3][i]):
            flag.append(1)#相同的可以
            id_use.append(pre_train_label[0][i])#将可以用的id保存起来
        else:
            flag.append(0)#不相同
            id_unuse.append(pre_train_label[0][i])
            id_idx_unuse.append(i)

    df.drop(df.index[id_idx_unuse], inplace=True)#删除没用的行
    df.drop(2,axis=1,inplace=True) 
    df.drop(3,axis=1,inplace=True) #有三个数据选其中一个数据

    df2 = df[1].str.split(',',expand=True)
    pre_image_label = df2[[1]]
    pre_text_label = df2[[0]]
    i = np.array(pre_image_label).tolist()
    t = np.array(pre_text_label).tolist()
    for j in range(1,len(i)):

        image_label.append(i[j][0])

    for k in range(1,len(t)):
        text_label.append(t[k][0])

   # print(text_label)

    return text_label,image_label,id_use


def convert_label(text_label,image_lable,id_use):
    #positive=[1,0],neutral=[0,0],negative=[0,1]
    convert_text_label=[]
    convert_image_lable=[]
    for i in text_label:
        if i == 'positive':
            convert_text_label.append([1,0,0])
        elif i == 'neutral':
            convert_text_label.append([0,1,0])
        elif i == 'negative':
            convert_text_label.append([0,0,1])
    for j in image_lable:
        if j == 'positive':
            convert_image_lable.append([1,0,0])
        elif j == 'neutral':
            convert_image_lable.append([0,1,0])
        elif j == 'negative':
           convert_image_lable.append([0,0,1])

    print(convert_image_lable[1:10])
    print(convert_text_label[1:10])

    return convert_text_label,convert_image_lable,id_use
    


def get_train_data(data_dir_path,id_use):
    text = []
    image = []

    for id in id_use:
        id = id + '.txt'
        domain = os.path.abspath(data_dir_path)#获取文件夹的路径
        info = os.path.join(domain,id)#将路径与文件名结合起来就是每个文件的完整路径
        info = open(info,'r',encoding='utf-8')#读取文件内容
        text.append(info.readline())#使用readline函数得到一条一条的信息，如果使用read获取全部信息亦可；
        info.close()

    for id in id_use:
        id = id + '.jpg'
        domain = os.path.abspath(data_dir_path)#获取文件夹的路径

        info = os.path.join(domain,id)#将路径与文件名结合起来就是每个文件的完整路径
        img = np.array(Image.open(info))#读取图片内容
        image.append(img)
    image = np.array(image)
    return image,text #text是一个文字数组，image是np存储的数组


def convert_example_to_feature(text):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []

    for sentence in text:
        bert_input = get_bert_input_V2(sentence,max_sentence_length)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
    return input_ids_list,token_type_ids_list,attention_mask_list

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels = 3)


def bert_model():

    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels = 3)
    model.get
    
    

    

    
    
    print(model.get_output_embeddings().shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-08, clipnorm=1)

    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_val_encoded)




##########################################################卷积神经网络部分##########################################


def conv(input,name,kh,kw,n_out,dh, dw):#dh,dw为卷积核kernel的stride，kh和kw为卷积核kernel的长宽，n_out为卷积核的厚度，也是输出下一层的厚度
    n_in = input.get_shape()[-1].value 
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(scope+"w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32, 
                                 initializer=tf.initializers.GlorotNorma(shape=shape))
                                 
        conv = tf.nn.conv2d(input, kernel, (1, dh, dw, 1), padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True,name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        return activation

def linear(input,name,n_out):#全连接操作,n_out为输出结点的个数
    n_in = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32, 
                                 initializer=tf.initializers.GlorotNorma(shape=shape))
                                
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu(tf.matmul(input, kernel) + biases, name=scope)
        return activation

        
def maxpooling(input, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


def cnn_model(input,keep_prob):#keep_prob为dropout设置的参数，可以为一个全局变量
    with tf.variable_scope('conv_1_1'):
        conv1_1 = conv(input, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1)
    with tf.variable_scope('conv_1_2'):
        conv1_2 = conv(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
    with tf.variable_scope('pool_1'):
        pool1 = maxpooling(conv1_2,   name="pool1",kh=2, kw=2, dw=2, dh=2)

    
    with tf.variable_scope('conv_2_1'):
        conv2_1 = conv(pool1,name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1)
    with tf.variable_scope('conv_2_2'):
        conv2_2 = conv(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1)
    with tf.variable_scope('pool_2'):
        pool2 = maxpooling(conv2_2,   name="pool2",kh=2, kw=2, dh=2, dw=2)


    with tf.variable_scope('conv_3_1'):
        conv3_1 = conv(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1)
    with tf.variable_scope('conv_3_2'):
        conv3_2 = conv(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
    with tf.variable_scope('conv_3_3'):
        conv3_3 = conv(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1)    
    with tf.variable_scope('pool_3'):
        pool3 = maxpooling(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    with tf.variable_scope('conv_4_1'):
        conv4_1 = conv(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
    with tf.variable_scope('conv_4_2'):
        conv4_2 = conv(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    with tf.variable_scope('conv_4_3'):
        conv4_3 = conv(conv4_2,  name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
    with tf.variable_scope('pool_4'):
        pool4 = maxpooling(conv4_3,name="pool4", kh=2, kw=2, dh=2, dw=2)

    with tf.variable_scope('conv_5_1'):
        conv5_1 = conv(pool4,name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
    with tf.variable_scope('conv_5_2'):
        conv5_2 = conv(conv5_1,  name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    with tf.variable_scope('conv_5_3'):
        conv5_3 = conv(conv5_2,  name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
    with tf.variable_scope('pool_5'):
        pool5 = maxpooling(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)



    with tf.variable_scope('gap'):#根据论文实现GAP替代全连接层实现分类
        GAP = tf.nn.avg_pool2d(pool5 ,ksize=[1,pool5.get_shape().as_list()[1],pool5.get_shape().as_list()[1],1],strides=[1,1,1,1],padding='VALID',name='GAP')

    with tf.variable_scope('fully_connect_1'):#全连接层1实现512到100
        linear1 = linear(GAP, name="linear1", n_out=100)
        linear1_drop = tf.nn.dropout(linear1, keep_prob, name="linear1_drop")

    with tf.variable_scope('fully_connect_2'):#全连接层2实现100到3，实现情绪的分类
        linear2 = linear(linear1_drop, name="linear2", n_out=3)
        linear2_drop = tf.nn.dropout(linear2, keep_prob, name="linear2_drop")


text_label,image_label,id_use =  get_label(label_path)

get_train_data(data_dir_path,id_use)
    

        