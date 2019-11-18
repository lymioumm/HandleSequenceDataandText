import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchtext import data
from torchtext.datasets import IMDB
from torchtext.vocab import GloVe
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')

class IMDBCnn(nn.Module):
    def __init__(self,n_vocab,hidden_size,n_cat,bs=1,kernel_size=3,max_len=200):
        super().__init__()
        self.hidden_size = hidden_size
        self.bs = bs
        self.e = nn.Embedding(n_vocab,hidden_size)
        # self.cnn_N = nn.Conv1d(2*max_len,max_len,kernel_size)
        self.cnn = nn.Conv1d(max_len,hidden_size,kernel_size)
        self.avg = nn.AdaptiveAvgPool1d(10)
        self.fc = nn.Linear(1000,n_cat)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,inp):
        bs = inp.size()[0]
        if bs != self.bs:
            self.bs = bs
        e_out = self.e(inp)
        cnn_o = self.cnn(e_out)
        cnn_avg = self.avg(cnn_o)
        cnn_avg = cnn_avg.view(self.bs,-1)
        # fc = F.dropout(self.fc(cnn_avg),p=0.2)   # 结果显示p = 0.2时，训练结果好
        # return self.softmax(fc)
        return self.softmax(cnn_avg)      # 不加池化层时效果更好

def fit(epoch, model, data_loader, phase='training', volatile=False):
        if phase == 'training':
            model.train()
        if phase == 'validation':
            model.eval()
            volatile = True
        running_loss = 0.0
        running_correct = 0
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        enumerate_ = enumerate(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            text, target = batch.text, batch.label
            text, target = text.cuda(), target.cuda()
            if phase == 'training':
                optimizer.zero_grad()
            output = model(text)  # 这一步会进入模型IMDBRnn的前向传播函数
            loss = F.nll_loss(output, target)
            # loss = F.cross_entropy(output,target)   # 训练效果不如nll_loss函数
            running_loss += F.nll_loss(output, target, size_average=False).item()
            preds = output.data.max(dim=1, keepdim=True)[1]
            running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
            if phase == 'training':
                loss.backward()
                optimizer.step()
        loss = running_loss / len(data_loader.dataset)
        accuracy = 100. * running_correct / len(data_loader.dataset)
        print(
            f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
        return loss, accuracy


def manifest():
    # preparing the data
    # TEXT = data.Field(lower = True,fix_length = 200,batch_first = False)     # RuntimeError: Given groups=1, weight of size 100 200 3, expected input[200, 32, 100] to have 200 channels, but got 32 channels instead
    TEXT = data.Field(lower = True,fix_length = 200,batch_first = True)
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
    # model = IMDBCnn(n_vocab,n_hidden,3,bs=32)    # 这一步主要进行模型IMDBRnn的__init__
    model = IMDBCnn(n_vocab,n_hidden, n_cat=3, bs=32, kernel_size=3)
    model = model.cuda()
    train_losses, train_accuracy = [],[]
    val_loss,val_accurayc = [],[]
    for epoch in range(1,5):
        epoch_loss,epoch_accuracy = fit(epoch,model,train_iter,phase='training')
        val_epoch_loss,val_epoch_accuracy = fit(epoch,model,test_iter,phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accurayc.append(val_epoch_accuracy)




    pass




def main():
    manifest()

    pass

if __name__ == '__main__':
    main()