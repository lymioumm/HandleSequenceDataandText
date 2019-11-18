import os

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import optimizer
from torchtext import data
from torchtext.datasets import IMDB
from torchtext.vocab import GloVe
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')


def manifest():
    # preparing the data
    TEXT = data.Field(lower = True,fix_length = 200,batch_first = False)
    LABEL = data.Field(sequential=False)
    train,test = IMDB.splits(TEXT,LABEL)
    TEXT.build_vocab(train,vectors = GloVe(name='6B',dim = 300),max_size = 10000,min_freq = 10)
    LABEL.build_vocab(train)

    # creating batches
    train_iter,test_iter = data.BucketIterator.splits((train,test),batch_size=32,device = None)
    train_iter.repeat = False
    test_iter.repeat = False

    pass

class IMDBRnn(nn.Module):
    def __init__(self,n_vocab,hidden_size,n_cat,bs=1,n1=2):
        super().__init__()
        self.hidden_size = hidden_size    # 100
        self.bs =bs                       # 32
        self.n1 = n1                      # 2
        self.e = nn.Embedding(n_vocab,hidden_size)
        self.rnn = nn.LSTM(hidden_size,hidden_size,n1)  # LSTM(100, 100, num_layers=2)
        self.fc2 = nn.Linear(hidden_size,n_cat)
        self.softmax = nn.LogSoftmax(dim = -1)
        # p = self.softmax
    def forward(self,inp):
        # text = inp            # inp的size为torch.Size([200, 32])，   inp为按批次处理好了的text
        bs = inp.size()[1]    # bs取inp.size()[1],则为32
        if bs != self.bs:
            self.bs = bs
        e_out = self.e(inp)   # e_out的shape ：torch.Size([200, 32, 100])  e是embedding模型，e_out输出转换后的嵌入结果，其中100是嵌入维度
        h0 = c0 = Variable(e_out.data.new(*(self.n1,self.bs,self.hidden_size)).zero_())     # torch.Size([2, 32, 100])
        rnn_o,_ = self.rnn(e_out,(h0,c0))   # run_o 为tensor类型，_ 为二元组tuple类型，其中每一个元素又是tensor类型
        rnn_o = rnn_o[-1]
        fc2_ = self.fc2(rnn_o)
        fc = F.dropout(self.fc2(rnn_o),p=0.2)   # 随着概率p值得降低，损失降低，精度提高，而且validation的值变化的较快，甚至效果比train的好，
        return self.softmax(fc)

def fit(epoch,model,data_loader,phase = 'training',volatile =False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    enumerate_ = enumerate(data_loader)
    for batch_idx,batch in enumerate(data_loader):
        text,target = batch.text,batch.label
        text,target = text.cuda(),target.cuda()
        if phase == 'training':
            optimizer.zero_grad()
        output = model(text)         # 这一步会进入模型IMDBRnn的前向传播函数
        loss = F.nll_loss(output,target)
        # loss = F.cross_entropy(output,target)   # 训练效果不如nll_loss函数
        running_loss += F.nll_loss(output,target,size_average=False).item()
        preds = output.data.max(dim=1,keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100.* running_correct/len(data_loader.dataset)
    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss,accuracy


def achieve():
    # preparing the data
    TEXT = data.Field(lower = True,fix_length = 200,batch_first = False)
    LABEL = data.Field(sequential=False)
    train,test = IMDB.splits(TEXT,LABEL)
    TEXT.build_vocab(train,vectors = GloVe(name='6B',dim = 300),max_size = 10000,min_freq = 10)
    LABEL.build_vocab(train)

    # creating batches
    train_iter,test_iter = data.BucketIterator.splits((train,test),batch_size=32,device = None)
    train_iter.repeat = False
    test_iter.repeat = False

    n_vocab = len(TEXT.vocab)
    n_hidden = 100   # 初始为100
    model = IMDBRnn(n_vocab,n_hidden,3,bs=32)    # 这一步主要进行模型IMDBRnn的__init__
    model = model.cuda()
    train_losses,train_accuracy = [],[]
    val_losses,val_accuracy = [],[]
    for epoch in range(1,5):
        epoch_loss,epoch_accuracy = fit(epoch,model,train_iter,phase='training')
        val_epoch_loss,val_epoch_accuracy = fit(epoch,model,test_iter,phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

    pass



def main():
    achieve()

    pass

if __name__ == '__main__':
    main()