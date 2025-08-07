# %% [markdown]
# In this kernel we will go through a sentiment analysis on imdb dataset using LSTM.

import torch
from torch import nn



# from classifier.conf.readConfig import Config

# config = Config().config
max_length=500
no_layers = 2
embedding_dim = 64
output_dim = 1
hidden_dim = 256
num_hiddens, num_layers =  128, 1
n_filters,filter_sizes,output_dim,dropout=100,[3,4,5],10,0.5


config={
    'embedding_dim': 64,
    'hidden_layer': 128,
    'max_length': 500,
    'lstm_hidden_layer': 128,
    'nc': 1
}
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


class LSTMBlock(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)


    def forward(self, x):
        x = x.squeeze(1)
        x, (h_n, c_n) = self.lstm(x)
        return x, h_n, c_n


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


class LSTMModel(nn.Module):
    def __init__(self, embedding_matrix):
        super(LSTMModel, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.embedding_matrix = embedding_matrix

        self.lstm = LSTMBlock(config['embedding_dim'], config['lstm_hidden_layer'],
                              config['lstm_num_layers'], bidirectional=config['lstm_bidirectional'])

        self.times = 2 if config['lstm_bidirectional'] else 1
        self.head = nn.Sequential(
            nn.Linear(config['lstm_hidden_layer'] * self.times * config['lstm_num_layers'], config['nc'])
        )

    def forward(self, x):
        x = self.embedding_matrix(x)
        _, x, _ = self.lstm(x)
        x = x.permute(1, 0, 2)  # [num_layers * num_dire, b_s, h_s] => [b_s, n_l * n_d, h_s]
        encoding = x.reshape(x.shape[0], -1)
        x = self.head(encoding)
        return x


# %%
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
# %%
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# %%
# base_csv = '/home/hadoop-epai/dolphinfs_ssd_hadoop-epai/chenjunqing/online_projects/playground/data/IMDB Dataset.csv'
# df = pd.read_csv(base_csv)[:10000]
base_tsv='/home/hadoop-epai/dolphinfs_ssd_hadoop-epai/chenjunqing/online_projects/playground/data/SST-2/train.tsv'
df = pd.read_csv(base_tsv,sep='\t')[:10000]

df.head()

# %% [markdown]
# ### Splitting to train and test data

# %% [markdown]
# We will split data to train and test initially. Doing this on earlier stage allows to avoid data lekage.
# 

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


def tockenize(x_train,y_train,x_val,y_val):
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


# %%
x_train_pad,y_train,x_test_pad,y_test,vocab = tockenize(x_train,y_train,x_test,y_test)

# %%
print(f'Length of vocabulary is {len(vocab)}')

# %% [markdown]
# ### Analysing review length

# %%
# rev_len = [len(i) for i in x_train]
# pd.Series(rev_len).hist()
# plt.show()
# pd.Series(rev_len).describe()

# %% [markdown]
# Observations : <br>a) Mean review length = around 69.<br> b) minimum length of reviews is 2.<br>c)There are quite a few reviews that are extremely long, we can manually investigate them to check whether we need to include or exclude them from our analysis.

# %% [markdown]
# ### Padding

# %% [markdown]
# Now we will pad each of the sequence to max length 

# %%
# def padding_(sentences, seq_len):
#     features = np.zeros((len(sentences), seq_len),dtype=int)
#     for ii, review in enumerate(sentences):
#         if len(review) != 0:
#             features[ii, -len(review):] = np.array(review)[:seq_len]
#     return features

# %%
#we have very less number of reviews with length > 500.
# #So we will consideronly those below it.
# x_train_pad = padding_(x_train,500)
# x_test_pad = padding_(x_test,500)


# %% [markdown]
# ### Batching and loading as tensor

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

# %% [markdown]
# We need to add an embedding layer because there are less words in our vocabulary. It is massively inefficient to one-hot encode that many classes. So, instead of one-hot encoding, we can have an embedding layer and use that layer as a lookup table. You could train an embedding layer using Word2Vec, then load it here. But, it's fine to just make a new layer, using it for only dimensionality reduction, and let the network learn the weights.

# %% [markdown]
# ### Model

# %%
class SentimentRNN(nn.Module):
    def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5):
        super(SentimentRNN,self).__init__()
 
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
 
        self.no_layers = no_layers
        self.vocab_size = vocab_size
    
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True)
        
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
    
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()
        
    def forward(self,x):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        lstm_out, _ = self.lstm(embeds)
        
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out
        
        
        



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
no_layers = 2
vocab_size = len(vocab) + 1 #extra 1 for padding
embedding_dim = 64
output_dim = 1
hidden_dim = 256
embedding_dim, num_hiddens, num_layers = 50, 128, 1
n_filters,filter_sizes,output_dim,dropout=100,[3,4,5],10,0.5



# %%
no_layers = 2
vocab_size = len(vocab) + 1 #extra 1 for padding
embedding_dim = 100
output_dim = 1
hidden_dim = 256
embedding_dim, num_hiddens, num_layers = 50, 128, 1
n_filters,filter_sizes,output_dim,dropout=100,[3,4,5],10,0.5


# model = SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)


# CNN model
model = TextCNNModel(vocab_size)

#moving to gpu
model.to(device)

print(model)


# %% [markdown]
# ### Training

# %%
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
epochs = 2
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
    for inputs, labels in train_loader:
        
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
        print(loss)
 
    
        
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
        torch.save(model.state_dict(), '/home/hadoop-epai/dolphinfs_ssd_hadoop-epai/chenjunqing/online_projects/playground/outmodels/state_dict.pt')
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










# def evaluate_accuracy(data_iter, net):
#     acc_sum, n = 0.0, 0
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(train_iter):
#             X, y = batch.text, batch.label
#             X = X.permute(1, 0)
# #             print(y.data)
#             y.data.sub_(1)  
            
#             if isinstance(net, torch.nn.Module):
#                 net.eval() # 评估模式, 这会关闭dropout
#                 acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#                 net.train() # 改回训练模式
#             else: # 自定义的模型, 不考虑GPU
#                 if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
#                     # 将is_training设置成False
#                     acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
#                 else:
#                     acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
#             n += y.shape[0]
#     return acc_sum / n

# def train(train_iter, test_iter, net, loss, optimizer, num_epochs):
#     batch_count = 0
#     PATH='CNN.pth'
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
#         bestacc=0.0
#         allLoss=0.
#         for batch_idx, batch in enumerate(train_iter):
#             X, y = batch.text, batch.label 
#             X = X.permute(1, 0) # 8*100
#             y.data.sub_(1)  #X转置 y为啥要减1
#             y_hat = net(X)
# #             print(f'X is {X.shape}')
# #             print(f'y_hat is {y_hat}')
# #             print(f'y_hat is {y_hat.argmax(dim=1)}')
# #             print(f'y is {y}')
#             l = loss(y_hat, y)
            
# #             print(f'loss is {l}')
            
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#             train_l_sum += l.item()
#             allLoss+= l.item()
#             train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
#             n += y.shape[0]
#             batch_count += 1
#             if (batch_idx+1)%20==0:
#                 print('batch_idx %d, loss %.4f, train acc %.3f,time %.1f sec' % ( batch_idx+1,train_l_sum / batch_count, train_acc_sum / n,time.time() - start))
#                 if bestacc<(train_acc_sum / n) and (train_acc_sum / n) >0.7:
#                     bestacc=(train_acc_sum / n)
#                     torch.save(net.state_dict(), PATH) 
#                     print(f'model saved!bestacc is {bestacc}')
            
#         test_acc = evaluate_accuracy(test_iter, net)
#         print(f'ave loss is {allLoss/batch_count}')
#         print(
#             'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
#             % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
#                test_acc, time.time() - start))
        

# lr, num_epochs =0.01, 5
# allLoss=0.
# # PATH='CNN_RNN.pth'
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# loss = nn.CrossEntropyLoss()

# train(train_dataiter, vad_dataiter, model, loss, optimizer, num_epochs)
# # torch.save(model.state_dict(), PATH) 
