import torch
import numpy as np
import torch.nn as nn


def euclidean_dist(x, y):  # x:[n, d]  y:[m, d]
  xx = (x*x).sum(dim=1, keepdim=True)  # (n, 1)
  yy = (y*y).sum(dim=1, keepdim=True).transpose(0, 1)  # (1, m)
  xy = torch.mm(x, y.transpose(0, 1))  # (n, m)
  return xx - 2*xy + yy


class PN(nn.Module):  # CPN w/o Pairwise Contrast
  def __init__(self, backbone, aug_num):
    super(PN, self).__init__()
    self.feature = backbone()
    self.aug_num = aug_num

  # def cuda(self):
  #   self.feature = nn.DataParallel(self.feature, device_ids=[0, 1, 2, 3]).cuda()
  #   return self

  def forward(self, x, temp=5.):  # x:[batch, aug_num, C, H, W]
    batch = x.size(0)
    x = x.reshape(-1, *x.size()[-3:])  # [batch*aug_num, C, H, W]
    features = self.feature(x)  # [batch*aug_num, d]
    support = features.reshape(batch, self.aug_num, -1)[:, 0, :]  # [batch, d]
    targets = torch.from_numpy(np.repeat(range(batch), self.aug_num)).cuda()  # [batch*aug_num]
    logits = - euclidean_dist(features, support)
    loss = nn.CrossEntropyLoss()(logits/temp, targets)
    return loss

  def train_loop(self, epoch, train_loader, optimizer):
    batch_number = len(train_loader)
    print_freq = batch_number//10+1
    avg_loss = 0

    for i, x in enumerate(train_loader):
      optimizer.zero_grad()
      loss = self.forward(x.cuda())
      loss.backward()
      optimizer.step()
      avg_loss = avg_loss+loss.item()

      if (i+1)%print_freq==0 or (i+1)==batch_number:
        print('Epoch {:d} | Iterations {:d}/{:d} | Loss {:f}'.format(epoch, i+1, batch_number, avg_loss/float(i+1)))
    return avg_loss/batch_number