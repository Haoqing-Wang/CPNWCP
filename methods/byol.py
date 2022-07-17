import torch
import numpy as np
import torch.nn as nn


def regression_loss(x, y):
  x = nn.functional.normalize(x, dim=1)
  y = nn.functional.normalize(y, dim=1)
  return 2.-2.*(x*y).sum(dim=-1)


class BYOL(nn.Module):
  def __init__(self, backbone, projection_dim=128, momentum=0.996):
    super(BYOL, self).__init__()
    self.projection_dim = projection_dim
    self.momentum = momentum
    self.feature = backbone()
    self.n_features = self.feature.feat_dim
    self.projector = nn.Sequential(nn.Linear(self.n_features, self.n_features//2),
                                   nn.BatchNorm1d(self.n_features//2),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.n_features//2, self.projection_dim))
    self.predictor = nn.Sequential(nn.Linear(self.projection_dim, self.projection_dim),
                                   nn.BatchNorm1d(self.projection_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.projection_dim, self.projection_dim))

    self.feature_k = backbone()
    self.projector_k = nn.Sequential(nn.Linear(self.n_features, self.n_features//2),
                                     nn.BatchNorm1d(self.n_features//2),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.n_features//2, self.projection_dim))

    for param_q, param_k in zip(self.feature.parameters(), self.feature_k.parameters()):
      param_k.data.copy_(param_q.data)
      param_k.requires_grad = False

    for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
      param_k.data.copy_(param_q.data)
      param_k.requires_grad = False

  # def cuda(self):
  #   self.feature = nn.DataParallel(self.feature, device_ids=[0, 1, 2, 3]).cuda()
  #   self.projector = nn.DataParallel(self.projector, device_ids=[0, 1, 2, 3]).cuda()
  #   self.predictor = nn.DataParallel(self.predictor, device_ids=[0, 1, 2, 3]).cuda()
  #
  #   self.feature_k = nn.DataParallel(self.feature_k, device_ids=[0, 1, 2, 3]).cuda()
  #   self.projector_k = nn.DataParallel(self.projector_k, device_ids=[0, 1, 2, 3]).cuda()
  #   return self

  @torch.no_grad()
  def _momentum_update_key_encoder(self):
    """
    Momentum update of the key encoder
    """
    for param_q, param_k in zip(self.feature.parameters(), self.feature_k.parameters()):
      param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
      param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

  def forward(self, x):  # x:[batch, 2, C, H, W]
    v1 = x[:, 0, :, :, :]
    v2 = x[:, 1, :, :, :]

    z1 = self.predictor(self.projector(self.feature(v1)))
    z2 = self.predictor(self.projector(self.feature(v2)))
    with torch.no_grad():
      self._momentum_update_key_encoder()
      z1_k = self.projector_k(self.feature_k(v1))
      z2_k = self.projector_k(self.feature_k(v2))

    loss = (regression_loss(z1, z2_k) + regression_loss(z2, z1_k)).mean()
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
    return avg_loss/batch_number