import torch
import numpy as np
import os
from nltk import ngrams
from torch import nn, optim
from torch.optim import optimizer
from torchtext import data, datasets
from torchtext.vocab import GloVe
import torch.nn.functional as F
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')

def ngrams_():
    thor_review = 'the action scenes were top notch in this movie. Thor has never been ' \
                  'this epic in the MCU. He does some pretty epic sh*t in this movie and' \
                  ' he is definitely not under-powered anymore. Thor in unleashed in this,' \
                  ' I love that.'
    print(f'Converting text into character:\n{list(thor_review)}')
    print(f'Conveerting text into words:\n{thor_review.split()}')
    print(f'nltk_bigrams:\n{list(ngrams(thor_review.split(), 2))}')
    print(f'ntlk_trigrams:\n{list(ngrams(thor_review.split(), 3))}')
    print(f'ntlk_fougrams:\n{list(ngrams(thor_review.split(), 4))}')

    pass

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}        # 生成一个字典，字典内每个单词对应一个索引： {'the': 1, 'action': 2, 'scenes': 3, 'were': 4, 'top': 5, 'notch': 6, 'in': 7, 'this': 8, 'movie.': 9, 'Thor': 10, 'has': 11, 'never': 12, 'been': 13, 'epic': 14, 'MCU.': 15, 'He': 16, 'does': 17, 'some': 18, 'pretty': 19, 'sh*t': 20, 'movie': 21, 'and': 22, 'he': 23, 'is': 24, 'definitely': 25, 'not': 26, 'under-powered': 27, 'anymore.': 28, 'unleashed': 29, 'this,': 30, 'I': 31, 'love': 32, 'that.': 33}
        self.idx2word = []        # 生成一个字符串数组，是字典内单词的集合
        self.length = 0           # 字典或字符串的长度
    def add_word(self,word):
        if word not in self.idx2word:    # 将没有出现的过单词加入到字典及字符串中
            self.idx2word.append(word)
            self.word2idx[word] = self.length + 1  # 这一步同时将单词匹配索引加入字典
            self.length += 1                # 字典长度加1
        return self.word2idx[word]          # 返回的是当前单词在字典内的索引
    def __len__(self):
        return len(self.idx2word)
    def onehot_encoded(self,word):
        vec = np.zeros(self.length)         # 把所有元素归零
        vec[self.word2idx[word]] = 1        # 把索引所在处置为1
        return vec
def dic_():
    thor_review = 'the action scenes were top notch in this movie. Thor has never been ' \
                  'this epic in the MCU. He does some pretty epic sh*t in this movie and' \
                  ' he is definitely not under-powered anymore. Thor in unleashed in this,' \
                  ' I love that.'
    dic = Dictionary()
    for tok in thor_review.split():
        dic.add_word(tok)

    # Results of word2idx
    print(f'dic.word2idx:\n{dic.word2idx}')
    print(f'dic.add_word:\n{dic.add_word(tok)}')
    print(f'dic.idx2word:\n{dic.idx2word}')

    # One-hot representation of the word 'were' is as follows:
    were_repre = dic.onehot_encoded('were')
    print(f'were_repre:\n{were_repre}')

    pass

def IMDB_():
    TEXT = data.Field(lower = True,batch_first = True,fix_length = 20)
    LABEL = data.Field(sequential=False)
    # print(f'TEXT:\n{TEXT}')
    # print(f'LABEL:\n{LABEL}')

    # download IMDB datasets use torchtext.datasets
    train,test = datasets.IMDB.splits(TEXT,LABEL)
    print('train.fields\n',train.fields)
    print(f'vars(train[0]):\n{vars(train[0])}')
    # # train 的一些参数
    # dirname = train.dirname
    # print(dirname)
    # name = train.name
    # print(name)
    # urls = train.urls
    # print(urls)
    TEXT.build_vocab(train,vectors = GloVe(name = '6B',dim = 300),max_size = 10000,min_freq = 10)
    # text = TEXT.build_vocab(train,vectors = GloVe(name = '6B',dim = 300),max_size = 10000,min_freq = 10)
    # print(f'TEXT.build_vocab:\n{text}')     # 输出为None
    glove = GloVe(name = '6B',dim = 300)      # torch.Size([400000, 300])
    print(f'glove:\n{glove}')
    G_vectors = glove.vectors     # torch.Size([400000, 300])
    print(f'glove.vectors:\n{G_vectors}')
    LABEL.build_vocab(train)
    freqs = TEXT.vocab.freqs        # 统计出现的次数
    print(f'TEXT.vocab.freq:\n{freqs}')
    # print(f'TEXT.vocab.freq:\n{TEXT.vocab.freqs.type}')    # AttributeError: 'Counter' object has no attribute 'type'
    vectors = TEXT.vocab.vectors
    print(f'TEXT.vocab.vectors:\n{vectors}')
    stoi = TEXT.vocab.stoi
    print(f'TEXT.vocab.stoi:\n{stoi}')    # defaultdict(<bound method Vocab._default_unk_index of <torchtext.vocab.Vocab object at 0x2b6b209b9f60>>

    # device = -1 表示使用cpu,None为gpu
    # BucketIterator help in batching all the text and replacing the words with the index number of the words
    train_iter,test_iter = data.BucketIterator.splits((train,test),batch_size = 128,device = None ,shuffle = True)

    batch = next(iter(train_iter))
    iters = iter(train_iter)

    B_text = batch.text
    print(f'batch.text:\n{B_text}')
    B_label = batch.label
    print(f'batch.label:\n{B_label}')

    pass

class EmbNet(nn.Module):
    def __init__(self, emb_size, hidden_size1, hidden_size2=200):
        super().__init__()
        self.embedding = nn.Embedding(emb_size, hidden_size1)
        self.fc = nn.Linear(hidden_size2, 3)

    def forward(self, x):
        y = x.size(0)
        y1 = x.size(1)
        # y2 = x.size(2)   # IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
        embeds = self.embedding(x).view(x.size(0), -1)
        out = self.fc(embeds)
        D_F = F.log_softmax(out, dim=-1)
        return F.log_softmax(out, dim=-1)


def fit(epoch,model,data_loader,phase = 'training',volatile = False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for batch_idx,batch in enumerate(data_loader):
        text,target = batch.text,batch.label
        text,target = text.cuda(),target.cuda()
        if phase == 'training':
            optimizer.zero_grad()
        output = model(text)
        loss = F.nll_loss(output,target)
        # running_loss += F.nll_loss(output,target,size_average=False).data[0]  # IndexError: invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number
        running_loss += F.nll_loss(output,target,size_average=False).item()
        preds = output.data.max(dim = 1,keepdim = True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. * running_correct/len(data_loader.dataset)
    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss,accuracy

def achieve_():

    TEXT = data.Field(lower=True, batch_first=True, fix_length=20)
    LABEL = data.Field(sequential=False)
    train,test = datasets.IMDB.splits(TEXT,LABEL)
    TEXT.build_vocab(train,vectors = GloVe(name = '6B',dim = 300),max_size = 10000,min_freq = 10)
    LABEL.build_vocab(train)
    train_iter,test_iter = data.BucketIterator.splits((train,test),batch_size = 128,device = -1,shuffle = True)
    lens = len(TEXT.vocab.stoi)
    model = EmbNet(len(TEXT.vocab.stoi), 10).to(device)   # model 必须在构建字典后定义
    train_losses,train_accuracy = [],[]
    val_losses,val_accuracy = [],[]
    train_iter.repeat = False
    test_iter.repeat = False
    for epoch in range(1,10):
        epoch_loss,epoch_accuracy = fit(epoch,model,train_iter,phase='training')
        val_epoch_loss,val_epoch_accuracy = fit(epoch,model,test_iter,phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

    # plot the trainging and test loss
    plt.figure(1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo', label='training loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r', label='validation loss')
    plt.legend()
    plt.savefig('Text_loss.jpg')
    # plots the training and test accuracy
    plt.figure(2)
    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'bo', label='training accuracy')
    plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, 'r', label='val accuracy')
    plt.legend()
    plt.savefig('Text_accuracy')


def main():
    # dic_()
    # IMDB_()
    achieve_()
    pass

if __name__ == '__main__':
    main()


