import torch.nn as nn
import numpy as np
from abc import abstractmethod


class MetaTemplate(nn.Module):
  def __init__(self, model_func, n_way, n_support, flatten=True):
    super(MetaTemplate, self).__init__()
    self.n_way = n_way
    self.n_support = n_support
    self.n_query = 16
    self.feature = model_func(flatten=flatten)

  @abstractmethod
  def set_forward(self, x):
    pass

  @abstractmethod
  def set_forward_loss(self, x):
    pass

  def forward(self, x):
    out = self.feature.forward(x)
    return out

  def correct(self, x):
    scores, loss = self.set_forward_loss(x)
    y_query = np.repeat(range(self.n_way), self.n_query)

    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:, 0] == y_query)
    return float(top1_correct), len(y_query), loss.item()*len(y_query)

  def test_loop(self, test_loader):
    acc_all = []

    iter_num = len(test_loader)
    for i, (x, _) in enumerate(test_loader):
      self.n_query = x.size(1) - self.n_support
      correct_this, count_this, loss_this = self.correct(x)
      acc_all.append(correct_this/count_this*100)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('--- Test Acc = %4.2f%% +- %4.2f%% ---'%(acc_mean, 1.96*acc_std/np.sqrt(iter_num)))
    return acc_mean