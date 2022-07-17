import torch
import numpy as np
import torch.nn as nn


def euclidean_dist(x, y):  # x:[n, d]  y:[m, d]
  xx = (x*x).sum(dim=1, keepdim=True)  # (n, 1)
  yy = (y*y).sum(dim=1, keepdim=True).transpose(0, 1)  # (1, m)
  xy = torch.mm(x, y.transpose(0, 1))  # (n, m)
  return xx - 2*xy + yy


class CPN_JS(nn.Module):  # CPN + Jensen–Shannon Confidence Penalty
  def __init__(self, backbone, aug_num):
    super(CPN_JS, self).__init__()
    self.feature = backbone()
    self.aug_num = aug_num

  # def cuda(self):
  #   self.feature = nn.DataParallel(self.feature, device_ids=[0, 1, 2, 3]).cuda()
  #   return self

  def forward(self, x, temp=5.):  # x:[batch, aug_num, C, H, W]
    batch = x.size(0)
    x = x.transpose(0, 1).reshape(-1, *x.size()[-3:])  # [aug_num*batch, C, H, W]
    features = self.feature(x)  # [aug_num*batch, d]
    eudis = euclidean_dist(features, features)  # [aug_num*batch, aug_num*batch]
    logits = -eudis.reshape(self.aug_num*batch*self.aug_num, batch)  # [aug_num*batch*aug_num, batch]
    # Cross Entropy Loss
    targets = torch.from_numpy(np.repeat(range(batch), self.aug_num)).repeat(self.aug_num).cuda()  # [aug_num*batch*aug_num]
    ce_loss = nn.CrossEntropyLoss()(logits/temp, targets)
    # JS_Divergence Regularization
    probs = (logits/np.sqrt(features.size(1))).softmax(1)  # [N, C]
    target_probs = torch.ones(probs.size()).cuda()/batch  # [N, C]
    js_loss = 0.5*(probs*torch.log(2.*probs+1e-12)-(probs+target_probs)*torch.log(probs+target_probs)).sum(1).mean()

    loss = ce_loss + js_loss
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
    return avg_loss / batch_number