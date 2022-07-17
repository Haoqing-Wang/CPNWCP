import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate


class ProtoNet(MetaTemplate):  # Prototype-based Nearest-Neighbor Classifier
  def __init__(self, model_func, n_way, n_support):
    super(ProtoNet, self).__init__(model_func, n_way, n_support)
    self.loss_fn = nn.CrossEntropyLoss()
    self.method = 'ProtoNet'

  # def cuda(self):
  #   self.feature = nn.DataParallel(self.feature, device_ids=[0, 1, 2, 3]).cuda()
  #   return self

  def set_forward(self, x):
    x = x.cuda().reshape(-1, *x.size()[2:])
    z = self.feature(x)  # (n_way*(n_support+n_query), d)
    z = z.reshape(self.n_way, -1, z.size(1))
    z_proto = z[:, :self.n_support].mean(1)
    z_query = z[:, self.n_support:].reshape(self.n_way*self.n_query, -1)

    dists = euclidean_dist(z_query, z_proto)
    scores = -dists
    return scores

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return scores, loss


def euclidean_dist(x, y):  # x:[n, d]  y:[m, d]
  xx = (x*x).sum(dim=1, keepdim=True)  # (n, 1)
  yy = (y*y).sum(dim=1, keepdim=True).transpose(0, 1)  # (1, m)
  xy = torch.mm(x, y.transpose(0, 1))  # (n, m)
  return xx - 2*xy + yy