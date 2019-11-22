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
        self.hidden_size = hidden_size     # 100
        self.bs = bs                       # 32
        self.e = nn.Embedding(n_vocab,hidden_size)      # e = Embedding(10002, 100)
        # self.cnn_N = nn.Conv1d(2*max_len,max_len,kernel_size)
        self.cnn = nn.Conv1d(max_len,hidden_size,kernel_size)       # cnn = Conv1d(in_channels = 200, out_channels = 100, kernel_size=(3,), stride=(1,))
        self.avg = nn.AdaptiveAvgPool1d(10)           # avg = AdaptiveAvgPool1d(output_size=10)
        self.fc = nn.Linear(1000,n_cat)               # fc = Linear(in_features=1000, out_features=3, bias=True)
        self.softmax = nn.LogSoftmax(dim=-1)          # softmax = LogSoftmax()
    def forward(self,inp):
        bs = inp.size()[0]    # bs = 32
        # inp =
        # tensor([[   9,  200,   10,  ...,   76,   84, 8021],
        #         [   2,   78,  823,  ...,    1,    1,    1],
        #         [2696,   58,    6,  ...,    1,    1,    1],
        #         ...,
        #         [   0, 7069,    5,  ..., 5216,   60, 1305],
        #         [   2, 1031, 2157,  ...,    1,    1,    1],
        #         [   9,  200,   10,  ...,    1,    1,    1]], device='cuda:0')
        # inp.shape = torch.Size([32, 200])
        if bs != self.bs:
            self.bs = bs
        e_out = self.e(inp)       # e -out.shape = torch.Size([32, 200, 100])
        cnn_o = self.cnn(e_out)   # cnn_o.shape = torch.Size([32, 100, 98])
        cnn_avg = self.avg(cnn_o)   # cnn_avg.shape = torch.Size([32, 100, 10])     # 注意到这里的10，这是在模型中设定为10，        self.avg = nn.AdaptiveAvgPool1d(10)           # avg = AdaptiveAvgPool1d(output_size=10)
        cnn_avg = cnn_avg.view(self.bs,-1)    # cnn_avg.shape = torch.Size([32, 1000])
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
            output = model(text)  # 这一步会进入模型IMDBRnn的前向传播函数     # output.shape = torch.Size([32, 1000])
            loss = F.nll_loss(output, target)
            # loss = F.cross_entropy(output,target)   # 训练效果不如nll_loss函数
            running_loss += F.nll_loss(output, target, size_average=False).item()
            preds = output.data.max(dim=1, keepdim=True)[1]     # preds.shape = torch.Size([32, 1])    # preds获得的是每一行中最大值的所在列

            temp1 = target.data.view_as(preds)    #  同 target.data.view(preds.size())
            temp2 = target.data.view(preds.size())
            running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()       # 这个正确的数目是根据什么计算的？
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
    # LABEL = <torchtext.data.field.Field object at 0x2aef04f2fa20>
    # TEXT = <torchtext.data.field.Field object at 0x2aef09c059b0>
    train,test = IMDB.splits(TEXT,LABEL)
    # test = <torchtext.datasets.imdb.IMDB object at 0x2aef09c05f98>
    # train = <torchtext.datasets.imdb.IMDB object at 0x2aef09c05a20>
    TEXT.build_vocab(train,vectors = GloVe(name='6B',dim = 300),max_size = 10000,min_freq = 10)
    LABEL.build_vocab(train)
    # creating batches
    train_iter,test_iter = data.BucketIterator.splits((train,test),batch_size=32,device = None)
    # test_iter = <torchtext.data.iterator.BucketIterator object at 0x2aef7f900f60>
    # train_iter = <torchtext.data.iterator.BucketIterator object at 0x2aef7f900fd0>
    train_iter.repeat = False
    test_iter.repeat = False
    n_vocab = len(TEXT.vocab)      # n_vocab = 10002
    n_hidden = 100   # 初始为100    # n_hidden = 100
    # model = IMDBCnn(n_vocab,n_hidden,3,bs=32)    # 这一步主要进行模型IMDBRnn的__init__
    model = IMDBCnn(n_vocab,n_hidden, n_cat=3, bs=32, kernel_size=3)
    # IMDBCnn(
    #   (e): Embedding(10002, 100)
    #   (cnn): Conv1d(200, 100, kernel_size=(3,), stride=(1,))
    #   (avg): AdaptiveAvgPool1d(output_size=10)
    #   (fc): Linear(in_features=1000, out_features=3, bias=True)
    #   (softmax): LogSoftmax()
    # )
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