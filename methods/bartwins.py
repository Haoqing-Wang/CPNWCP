import torch
import numpy as np
import torch.nn as nn


def criterion(z1, z2, lam):  # (N, d)
  N, d = z1.size()
  z1_norm = (z1-z1.mean(0, keepdim=True))/z1.std(0, keepdim=True)
  z2_norm = (z2-z2.mean(0, keepdim=True))/z2.std(0, keepdim=True)
  C = torch.mm(z1_norm.transpose(0, 1), z2_norm)/N

  I = torch.eye(d).cuda()
  mask = lam * (1. - I) + I
  C_diff = (C - I).pow(2)
  loss = (mask * C_diff).sum()
  return loss


class BarTwins(nn.Module):
  def __init__(self, backbone):
    super(BarTwins, self).__init__()
    self.feature = backbone()
    self.n_features = self.feature.feat_dim
    self.projector = nn.Sequential(nn.Linear(self.n_features, self.n_features//2),
                                   nn.ReLU(),
                                   nn.Linear(self.n_features//2, 1024))

  # def cuda(self):
  #   self.feature = nn.DataParallel(self.feature, device_ids=[0, 1, 2, 3]).cuda()
  #   self.projector = nn.DataParallel(self.projector, device_ids=[0, 1, 2, 3]).cuda()
  #   return self

  def forward(self, x):  # x:[batch, 2, C, H, W]
    v1 = x[:, 0, :, :, :]
    v2 = x[:, 1, :, :, :]

    z1 = self.feature(v1)
    z1 = self.projector(z1)
    z1 = nn.functional.normalize(z1, dim=1)

    z2 = self.feature(v2)
    z2 = self.projector(z2)
    z2 = nn.functional.normalize(z2, dim=1)

    loss = criterion(z1, z2, 1e-2)
    return loss

  def train_loop(self, epoch, train_loader, optimizer):
    batch_number = len(train_loader)
    print_freq = batch_number//10+1
    avg_loss = 0.

    for i, x in enumerate(train_loader):
      optimizer.zero_grad()
      loss = self.forward(x.cuda())
      loss.backward()
      optimizer.step()
      avg_loss = avg_loss+loss.item()

      if (i+1)%print_freq==0 or (i+1)==batch_number:
        print('Epoch {:d} | Iterations {:d}/{:d} | Loss {:f}'.format(epoch, i+1, batch_number, avg_loss/float(i+1)))
    return avg_loss / batch_number