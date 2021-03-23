from pytorch_transformers import BertTokenizer #该类实现数据的token化
from pytorch_transformers import BertModel
import torch
import torch.nn as nn
import torch.functional as F
import scipy.misc
import os
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
from PIL import ImageFile


label_path = 'D:\code\\bert\word_data2\labelResultAll.txt'
img_dir_path = 'D:\code\\bert\word_data2\data\pic'
text_dir_path = 'D:\code\\bert\word_data2\data'
max_length = 120
number_of_epochs = 8 #训练循环次数
learning_rate = 2e-5 #学习率
batch_size = 50 #一次输入的数量
epochs = 5#循环多少轮
keep_prob = 0.5 
beta1 = 0.5


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

def get_bert_input_V2(sentence):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')#加载bert实现序列化预训练模型
    bert_input = tokenizer.encode_plus(sentence,add_special_tokens = True,#增加句子的前后缀
                            max_length=max_length,
                            pad_to_max_length = True,#是否填充到一致长度
                            return_attention_mask = True)#实现attention_mask
    #print(bert_input)
    return bert_input



###########################################数据集处理模块################################################


def resize_image(id_use):
    # 获取输入文件夹中的所有文件
    files = os.listdir("D:\code\\bert\word_data2\data")

    output_dir = "D:\code\\bert\word_data2\data\pic"
    # 判断输出文件夹是否存在，不存在则创建
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in id_use:
        img = Image.open("D:\code\\bert\word_data2\data\\"+ file + '.jpg')
        if img.mode == "P":
            img = img.convert('RGB')
        if img.mode == "RGBA":
            img = img.convert('RGB')
        img = img.resize((640, 480), Image.ANTIALIAS)
        img.save(os.path.join(output_dir, file +'.jpg'))

    print('图片输出完成')





def get_label(label_path):
    pre_train_label = pd.read_csv(label_path,sep="\t",header=None,encoding="gbk")
    df = pd.DataFrame(pre_train_label)
    text_label = []
    image_label = []
    real_id_use=[]

    flag = []
    id_use = []
    id_unuse = []
    id_idx_unuse = []

    df2 = df[1].str.split(',',expand=True)

    #遍历整个行，一共有19600条数据
    for i in range(1,19600):
        if(pre_train_label[1][i] == pre_train_label[2][i] or pre_train_label[1][i] == pre_train_label[3][i] or pre_train_label[2][i] == pre_train_label[3][i]) :
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
    print('获取标签成功')

    return text_label,image_label,id_use


def convert_label(text_label,image_label,id_use):
    #positive=[1,0],neutral=[0,0],negative=[0,1]
    convert_text_label=[]
    convert_image_label=[]
    for i in text_label:
        if i == 'positive':
            convert_text_label.append([1,0,0])
        elif i == 'neutral':
            convert_text_label.append([0,1,0])
        elif i == 'negative':
            convert_text_label.append([0,0,1])
    for j in image_label:
        if j == 'positive':
            convert_image_label.append([1,0,0])
        elif j == 'neutral':
            convert_image_label.append([0,1,0])
        elif j == 'negative':
           convert_image_label.append([0,0,1])

    #print(convert_image_label[1:10])
   # print(convert_text_label[1:10])

    return convert_text_label,convert_image_label,id_use
    


def get_train_data(img_dir_path,id_use,text_dir_path):
    text = []
    image = []

    for id in id_use:
        id = id + '.txt'
        domain = os.path.abspath(text_dir_path)#获取文件夹的路径
        info = os.path.join(domain,id)#将路径与文件名结合起来就是每个文件的完整路径
        info = open(info,'r',encoding='utf-8')#读取文件内容
        text.append(info.readline())#使用readline函数得到一条一条的信息，如果使用read获取全部信息亦可；
        info.close()

    for id in id_use:
        id = id + '.jpg'
        domain = os.path.abspath(img_dir_path)#获取文件夹的路径

        info = os.path.join(domain,id)#将路径与文件名结合起来就是每个文件的完整路径
        img = np.array(Image.open(info))#读取图片内容
        image.append(img)
    return image,text #text是一个文字数组，image是np存储的数组

def convert_example_to_feature(text):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    counter = 0

    for sentence in text:
        bert_input = get_bert_input_V2(sentence)
        
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
    return input_ids_list,token_type_ids_list,attention_mask_list



def map_example_to_dict(input_ids, attention_masks, token_type_ids):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }




def bert_model(text,text_label):

    input_ids_list,token_type_ids_list,attention_mask_list = convert_example_to_feature(text)#转化文档变成bert_模型需要输入的格式
    input_ids_list = np.array(input_ids_list).astype(np.int32)
    token_type_ids_list = np.array(token_type_ids_list).astype(np.int32)
    attention_mask_list = np.array(attention_mask_list).astype(np.int32)

    print('转化完成')

    text_data_train = map_example_to_dict(input_ids_list,token_type_ids_list,attention_mask_list)

    model = BertModel.from_pretrained('bert-base-uncased')

    bert_output = model(text_data_train)[1]
    bert_output = bert_output.numpy()
    print(bert_output)
    return(bert_output)

class VGG16(nn.Module):
   
    def __init__(self):
        super(VGG16, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3,64,3)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1)) 
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))
        self.conv2_1 = nn.Conv2d(64, 128, 3)

        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1)) # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 128 * 56 * 56
            
        self.conv3_1 = nn.Conv2d(128, 256, 3) # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 256 * 28 * 28
            
        self.conv4_1 = nn.Conv2d(256, 512, 3) # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 14 * 14
            
        self.conv5_1 = nn.Conv2d(512, 512, 3) # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 7 * 7

        self.GAP = nn.AvgPool2d(kernel_size=[7,7],stride=[1,1,1,1])

        self.linear_3= nn.Linear(512,100)
        self.linear_4= nn.Linear(100,50)

        self.linear_1 = nn.Linear(768,100)
        self.linear_2 = nn.Linear(768,50)
        self.linear_5 = nn.Linear(50,3)

    def forward(self,image,text):
        in_size = image.size(0)
        out = self.conv1_1(image)
        out = F.relu(out)
        out = self.conv1_2(out) # 222
        out = F.relu(out)
        out = self.maxpool1(out) # 112
        
        out = self.conv2_1(out) # 110
        out = F.relu(out)
        out = self.conv2_2(out) # 110
        out = F.relu(out)
        out = self.maxpool2(out) # 56
        
        out = self.conv3_1(out) # 54
        out = F.relu(out)
        out = self.conv3_2(out) # 54
        out = F.relu(out)
        out = self.conv3_3(out) # 54
        out = F.relu(out)
        out = self.maxpool3(out) # 28
        
        out = self.conv4_1(out) # 26
        out = F.relu(out)
        out = self.conv4_2(out) # 26
        out = F.relu(out)
        out = self.conv4_3(out) # 26
        out = F.relu(out)
        out = self.maxpool4(out) # 14
        
        out = self.conv5_1(out) # 12
        out = F.relu(out)
        out = self.conv5_2(out) # 12
        out = F.relu(out)
        out = self.conv5_3(out) # 12
        out = F.relu(out)
        out = self.maxpool5(out) # 7
        out = self.GAP(out)
        out = F.relu(out)
        out = self.linear_3(out)
        out = F.relu(out)
        
        out_1 = self.linear_1(text)
        out_1 = F.relu(out_1)
        out = out + out_1
        out = F.relu(out)
        out = self.linear_4(out)
        out = F.relu(out)
        out_2 = self.linear_2(text)
        out_2 = F.relu(out_2)
        out = out + out_2
        out = self.linear_5(out)
        out = F.log_softmax(out, dim=1)
        return out

model = VGG16()
loss_function = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=0.001)


def train():
    text_label,image_label,id_use =  get_label(label_path)
    text_label,image_label,id_use = convert_label(text_label,image_label,id_use)

    img , text = get_train_data(img_dir_path,id_use,text_dir_path)#得到需要训练的文档，和图片
    print('得到需要训练的文档，和图片')

    pic_list= glob(os.path.join(img_dir_path,"*.jpg"))
    print('读取图片列表完成')
    batch_idxs = len(pic_list)/batch_size
    for epoch in range(0,epochs):

        for idxs in range(0,int(batch_idxs)):

            batch_files= pic_list[idxs*batch_size:(idxs +1)*batch_size]
        
            batch_img_files=[scipy.misc.imread(batch_file).astype(np.float) for batch_file in batch_files]
            batch_images = np.array(batch_img_files).astype(np.float32)

            batch_label_files = text_label[idxs*batch_size:(idxs +1)*batch_size]
            batch_label = np.array(batch_label_files).astype(np.float32)
            batch_text_files = text[idxs*batch_size:(idxs +1)*batch_size]
            
            batch_output = bert_model(batch_text_files,batch_label)
            
            softmax = model(batch_output,batch_images)
            loss= loss_function(softmax,batch_label)
            opt.zero_grad() 
            loss.backward()
            opt.step()

train()


            





    






