
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



# 数据处理
X,y = df['review'].values,df['sentiment'].values
x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)
print(f'shape of train data is {x_train.shape}')
print(f'shape of test data is {x_test.shape}')


# 
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

# ### 分词

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
    print("Vocabulary saved")

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



x_train_pad,y_train,x_test_pad,y_test,vocab = tockenize(x_train,y_train,x_test,y_test)

print(f'Length of vocabulary is {len(vocab)}')




# %%
# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

# dataloaders
batch_size = 50

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

# %%
# obtain one batch of training data
train_dataiter = iter(train_loader)
vad_dataiter = iter(valid_loader)



sample_x, sample_y = next(train_dataiter)

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print('Sample input: \n', sample_y)





# 计算词表大小
vocab_size = len(vocab) + 1 #extra 1 for padding



# CNN model
model = TextCNNModel(vocab_size)
# LSTM model
model = LSTMModel()

#moving to gpu
model.to(device)


# 打印模型结构
print(model)


# ### Training
# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# function to predict accuracy
def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()


# %%
clip = 5
epochs = 10
valid_loss_min = np.inf
# train for some number of epochs
epoch_tr_loss,epoch_vl_loss = [],[]
epoch_tr_acc,epoch_vl_acc = [],[]

for epoch in range(epochs):
    train_losses = []
    train_acc = 0.0
    model.train()
    # initialize hidden state 
    # h = model.init_hidden(batch_size)
    for inputs, labels in tqdm(train_loader):
        
        inputs, labels = inputs.to(device), labels.to(device)   
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        # h = tuple([each.data for each in h])
        
        model.zero_grad()
        output = model(inputs)
        
        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        train_losses.append(loss.item())
        # calculating accuracy
        accuracy = acc(output,labels)
        train_acc += accuracy
        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        # print(loss)
 
    
        
    # val_h = model.init_hidden(batch_size)
    val_losses = []
    val_acc = 0.0
    model.eval()
    for inputs, labels in valid_loader:
            # val_h = tuple([each.data for each in val_h])

            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)
            val_loss = criterion(output.squeeze(), labels.float())

            val_losses.append(val_loss.item())
            
            accuracy = acc(output,labels)
            val_acc += accuracy
            
    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = train_acc/len(train_loader.dataset)
    epoch_val_acc = val_acc/len(valid_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(f'Epoch {epoch+1}') 
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
    if epoch_val_loss <= valid_loss_min:
        torch.save(model, '/home/hadoop-epai/dolphinfs_ssd_hadoop-epai/chenjunqing/online_projects/playground/outmodels/state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
        valid_loss_min = epoch_val_loss
    print(25*'==')
    

# %%
fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 2, 1)
plt.plot(epoch_tr_acc, label='Train Acc')
plt.plot(epoch_vl_acc, label='Validation Acc')
plt.title("Accuracy")
plt.legend()
plt.grid()
    
plt.subplot(1, 2, 2)
plt.plot(epoch_tr_loss, label='Train loss')
plt.plot(epoch_vl_loss, label='Validation loss')
plt.title("Loss")
plt.legend()
plt.grid()

plt.show()


# %% [markdown]
# ### Inferance

# %%
def predict_text(text):
        word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() 
                         if preprocess_string(word) in vocab.keys()])
        word_seq = np.expand_dims(word_seq,axis=0)
        pad =  torch.from_numpy(padding_(word_seq,500))
        inputs = pad.to(device)
        batch_size = 1
        # h = model.init_hidden(batch_size)
        # h = tuple([each.data for each in h])
        output = model(inputs)
        return(output.item())

# %%

index = 30
print(df['review'][index])
print('='*70)
print(f'Actual sentiment is  : {df["sentiment"][index]}')
print('='*70)
pro = predict_text(df['review'][index])
status = "positive" if pro > 0.5 else "negative"
pro = (1 - pro) if status == "negative" else pro
print(f'Predicted sentiment is {status} with a probability of {pro}')

# %%

index = 32
print(df['review'][index])
print('='*70)
print(f'Actual sentiment is  : {df["sentiment"][index]}')
print('='*70)
pro = predict_text(df['review'][index])
status = "positive" if pro > 0.5 else "negative"
pro = (1 - pro) if status == "negative" else pro
print(f'predicted sentiment is {status} with a probability of {pro}')








