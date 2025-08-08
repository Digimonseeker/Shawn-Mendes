
import torch
from torch import nn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
# from nltk.corpus import stopwords 
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time

#首先看GPU是否可以使用
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


# max_length=500
# no_layers = 2
# embedding_dim = 64
# output_dim = 1
# hidden_dim = 256
# num_hiddens, num_layers =  128, 1
# n_filters,filter_sizes,output_dim,dropout=100,[3,4,5],10,0.5

# 定义模型参数
config={
    'embedding_dim': 64,
    'hidden_layer': 128,
    'max_length': 500,
    'lstm_hidden_layer': 128,
    'lstm_num_layers':2,
    'lstm_bidirectional': True,
    'nc': 1
}



#lstm模块
class LSTMBlock(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)


    def forward(self, x):
        x = x.squeeze(1)
        x, (h_n, c_n) = self.lstm(x)
        return x, h_n, c_n

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.embedding_matrix = nn.Embedding(vocab_size, config['embedding_dim'])


        self.lstm = LSTMBlock(config['embedding_dim'], config['lstm_hidden_layer'],
                              config['lstm_num_layers'], bidirectional=config['lstm_bidirectional'])

        self.times = 2 if config['lstm_bidirectional'] else 1
        self.head = nn.Sequential(
            nn.Linear(config['lstm_hidden_layer'] * self.times * config['lstm_num_layers'], config['nc'])
        )
        self.sig = nn.Sigmoid()


    def forward(self, x):
        x = self.embedding_matrix(x)
        _, x, _ = self.lstm(x)
        x = x.permute(1, 0, 2)  # [num_layers * num_dire, b_s, h_s] => [b_s, n_l * n_d, h_s]
        encoding = x.reshape(x.shape[0], -1)
        x = self.sig(self.head(encoding))
        return x





# 卷积模块
class CNNBlock(nn.Module):
    def __init__(self, kernel_size, out_channels=config['hidden_layer']):
        super(CNNBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_size, config['embedding_dim'])),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool1d(kernel_size=(config['max_length'] - kernel_size + 1))
        self.avepool = nn.AvgPool1d(kernel_size=(config['max_length'] - kernel_size + 1))
        

    def forward(self, bath_embedding):
        x = self.sequential(bath_embedding)
        x = x.squeeze(-1)   # 删掉最后一个维度
        x_maxpool = self.maxpool(x)
        x_maxpool = x_maxpool.squeeze(-1)  # 删掉最后一个维度
        # x_avepool = self.avepool(x)
        # x_avepool = x_avepool.squeeze(-1)
        # concat_x = torch.cat((x_maxpool, x_avepool), dim=1)
        # return concat_x
        return x_maxpool

# 文本cnn模块
class TextCNNBlock(nn.Module):

    def __init__(self):
        super(TextCNNBlock, self).__init__()
        # 特征提取层
        self.b0 = CNNBlock(5)
        self.b1 = CNNBlock(7)
        self.b2 = CNNBlock(11)
        # self.b3 = CNNBlock(17)
        # self.b4 = CNNBlock(25)
        # self.b5 = CNNBlock(3)

        # 特征融合部分 及 输出头
        self.neck = nn.Sequential(
            nn.ReLU(),
            nn.Linear(3 * config['hidden_layer'], 512)
        )
        self.dropout = nn.Dropout(0.5)


    def forward(self, x, label=None):
        x=x.unsqueeze(1)

        b0_results = self.b0(x)
        b1_results = self.b1(x)
        b2_results = self.b2(x)
        # b3_results = self.b3(x)
        # b4_results = self.b4(x)
        # b5_results = self.b5(x)

        features = torch.cat([b0_results, b1_results, b2_results], dim=1)
        output = self.dropout(self.neck(features))
        return output


class TextCNNModel(nn.Module):
    def __init__(self, vocab_size):
        super(TextCNNModel, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.embedding_matrix = nn.Embedding(vocab_size, config['embedding_dim'])
        self.textCNN = TextCNNBlock()
        self.head = nn.Sequential(
            nn.Linear(512, config['nc'])
        )
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.embedding_matrix(x)
        x = self.textCNN(x)


        x = self.sig(self.dropout(self.head(x)))
        return x



# 读取情感分类文件
base_csv = '/home/hadoop-epai/dolphinfs_ssd_hadoop-epai/chenjunqing/online_projects/playground/data/IMDB Dataset.csv'
df = pd.read_csv(base_csv)[:10000]
# base_tsv='/home/hadoop-epai/dolphinfs_ssd_hadoop-epai/chenjunqing/online_projects/playground/data/SST-2/train.tsv'
# df = pd.read_csv(base_tsv,sep='\t')[:10000]



# %%
X,y = df['review'].values,df['sentiment'].values
x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)
print(f'shape of train data is {x_train.shape}')
print(f'shape of test data is {x_test.shape}')

# %% [markdown]
# ### Analysing sentiment

# %%
dd = pd.Series(y_train).value_counts()
sns.barplot(x=np.array(['negative','positive']),y=dd.values)
plt.show()

# %%
def get_stop_words():
    file_object = open('/home/hadoop-epai/dolphinfs_ssd_hadoop-epai/chenjunqing/online_projects/playground/data/stopwords.txt',encoding='utf-8')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words

# %% [markdown]
# ### Tockenization

# %%
def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def tockenize(x_train,y_train,x_val,y_val,save_vocab_path='vocab.pt'):
    word_list = []

    stop_words = get_stop_words()
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)
  
    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

    # 保存词汇表到文件
    torch.save(onehot_dict, '/home/hadoop-epai/dolphinfs_ssd_hadoop-epai/chenjunqing/online_projects/playground/outmodels/vocab.pt')
    print(f"Vocabulary saved")

    # tockenize
    final_list_train,final_list_test = [],[]
    for sent in x_train:
            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                     if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
            final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                    if preprocess_string(word) in onehot_dict.keys()])
            
    encoded_train = [1 if label =='positive' else 0 for label in y_train]  
    encoded_test = [1 if label =='positive' else 0 for label in y_val] 

    x_train_pad = padding_(final_list_train,500)
    x_test_pad = padding_(final_list_test,500)


    return x_train_pad, np.array(encoded_train),x_test_pad, np.array(encoded_test),onehot_dict


# # %%
# x_train_pad,y_train,x_test_pad,y_test,vocab = tockenize(x_train,y_train,x_test,y_test)

# # %%
# print(f'Length of vocabulary is {len(vocab)}')







import torch.nn as nn
import torch.nn.functional as F

class FastText(nn.Module):
    def __init__(self, n_vocab, embed, hidden_size,num_classes,dropout,n_gram_vocab):
        super(FastText, self).__init__()
#         if embedding_pretrained is not None:
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=False)
#         else:
#             self.embedding = nn.Embedding(n_vocab, embed, padding_idx=config.n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(n_gram_vocab, embed)
        self.embedding_ngram3 = nn.Embedding(n_gram_vocab, embed)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed * 3, hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        out_word = self.embedding(x[0])
        print(f"out_printt {out_word.shape}{out_word}")
        out_bigram = self.embedding_ngram2(x[2])
        print(f"out_bigram {out_bigram.shape}{out_bigram}")
        out_trigram = self.embedding_ngram3(x[3])
        print(f"out_trigram {out_trigram.shape}{out_trigram}")
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

              


import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        
        self.conv_0 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[0], 10)
                               )
#         
        self.conv_1 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[1], 10))
        
        self.conv_2 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[2], 10))
        self.pool0 =  nn.MaxPool2d((2, 2), stride=(1, 1))
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
        print('embedding',embedded.shape)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        print('embedding',embedded.shape)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved_0 = F.relu(self.conv_0(embedded)) #embedded = [batch size, 1, sent len]
        conved_1 = F.relu(self.conv_1(embedded))
        conved_2 = F.relu(self.conv_2(embedded))
        print('conved_0',conved_0.shape)
        print('conved_1',conved_1.shape)
        print('conved_2',conved_2.shape)
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
#         pooled_0 = F.max_pool2d(conved_0, conved_0.shape[2]).squeeze(2) # embedded = [batch size, 100]
        pooled_0 = self.pool0(conved_0)
        print('pooled_0',pooled_0.shape)

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2) # embedded = [batch size, 100]
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        print('pooled_0',pooled_0.shape)
        print('pooled_1',pooled_1.shape)
        print('pooled_2',pooled_2.shape)
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))
        print('cat',cat.shape)
        

        #cat = [batch size, n_filters * len(filter_sizes)]

        
        return self.fc(cat)



# %%
vocab = torch.load('/home/hadoop-epai/dolphinfs_ssd_hadoop-epai/chenjunqing/online_projects/playground/outmodels/vocab.pt')  # 假设词汇表已保存
no_layers = 2
vocab_size = len(vocab) + 1 #extra 1 for padding
embedding_dim = 64
output_dim = 1
hidden_dim = 256
embedding_dim, num_hiddens, num_layers = 50, 128, 1
n_filters,filter_sizes,output_dim,dropout=100,[3,4,5],10,0.5



# CNN model
# model = TextCNNModel(vocab_size)
# LSTM model
model = LSTMModel()

#moving to gpu
model.to(device)


# 打印模型结构
print(model)




# # ### Training
# # loss and optimization functions
# lr=0.001

# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# function to predict accuracy
def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

# 加载保存的模型
def load_model(model_path, device='cpu'):
    model = torch.load(model_path, map_location=device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # state_dict = torch.load(model_path, map_location=device)
    # model.load_state_dict(state_dict)

    model.eval()  # 设置为评估模式
    return model


# %% [markdown]
# ### Inference

# 加载模型
# 1. 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 加载模型和词汇表
model = load_model('/home/hadoop-epai/dolphinfs_ssd_hadoop-epai/chenjunqing/online_projects/playground/outmodels/state_dict.pt', device)


# %%
# def predict_text(text):
#         word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() 
#                          if preprocess_string(word) in vocab.keys()])
#         word_seq = np.expand_dims(word_seq,axis=0)
#         pad =  torch.from_numpy(padding_(word_seq,500))
#         inputs = pad.to(device)
#         batch_size = 1
#         # h = model.init_hidden(batch_size)
#         # h = tuple([each.data for each in h])
#         output = model(inputs)
#         return(output.item())

# 预测函数
def predict_text(model, vocab, text, device='cpu', max_len=500):
    # 预处理文本并转换为索引序列
    word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() 
                         if preprocess_string(word) in vocab.keys()])
    
    # 添加批次维度并填充
    word_seq = np.expand_dims(word_seq, axis=0)
    pad = torch.from_numpy(padding_(word_seq, max_len))
    
    # 移动到设备
    inputs = pad.to(device)
    
    # 预测
    with torch.no_grad():  # 不计算梯度
        output = model(inputs)
    
    # 返回预测结果
    return output.item()


# %%


# 3. 要预测的句子
sentences = [
    "I love this movie, it's amazing!",
    "This product is terrible, I hate it.",
    "The service was okay, nothing special.",
    # 添加更多句子...
]

# 4. 预测每个句子的情感
for sent in sentences:
    prediction = predict_text(model, vocab, sent, device)
    sentiment = "positive" if prediction > 0.5 else "negative"
    print(f"Sentence: {sent}")
    print(f"Prediction score: {prediction:.4f} → {sentiment}\n")









