import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x, y):  # x:[n, d]  y:[m, d]
  xx = (x*x).sum(dim=1, keepdim=True)  # (n, 1)
  yy = (y*y).sum(dim=1, keepdim=True).transpose(0, 1)  # (1, m)
  xy = torch.mm(x, y.transpose(0, 1))  # (n, m)
  return xx - 2*xy + yy


def cosin_sim(x, y):  # x:[n, d]  y:[m, d]
  x = x.unsqueeze(1)  # [n, 1, d]
  y = y.unsqueeze(0)  # [1, m, d]
  return F.cosine_similarity(x, y, dim=-1)


class CPN_WCP(nn.Module):  # CPN + Wasserstein Confidence Penalty
  def __init__(self, backbone, aug_num, gamma):
    super(CPN_WCP, self).__init__()
    self.feature = backbone()
    self.aug_num = aug_num
    self.gamma = gamma

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
    # Wasserstein Distance Regularization
    probs = (logits/np.sqrt(features.size(1))).softmax(1)  # [N, C]
    target_probs = torch.ones(probs.size()).cuda()/batch  # [N, C]
    with torch.no_grad():
      features = features.reshape(self.aug_num, batch, -1).mean(0)  # [batch, d]
      cost = 1.-cosin_sim(features, features)  # [batch, batch]
      cost = (cost-cost.min(-1, keepdims=True)[0])/(cost.max(-1, keepdims=True)[0]-cost.min(-1, keepdims=True)[0])
      cost = (self.gamma*cost+torch.eye(batch).cuda()).unsqueeze(0).repeat(probs.size(0), 1, 1)
    wcp_loss = self.SinkhornDistance(probs, target_probs, cost)

    loss = ce_loss + wcp_loss
    return loss

  def M(self, C, u, v, eps):
    "Modified cost for logarithmic updates"
    return (-C+u.unsqueeze(-1)+v.unsqueeze(-2))/eps

  def SinkhornDistance(self, p1, p2, C, itr=5, eps=0.5):
    u = torch.zeros_like(p1)
    v = torch.zeros_like(p2)
    for _ in range(itr):
      u = eps*(torch.log(p1+1e-12)-torch.logsumexp(self.M(C, u, v, eps), dim=-1)) + u
      v = eps*(torch.log(p2+1e-12)-torch.logsumexp(self.M(C, u, v, eps).transpose(-2, -1), dim=-1)) + v

    pi = torch.exp(self.M(C, u, v, eps))
    return (pi*C).sum((-2, -1)).mean()

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