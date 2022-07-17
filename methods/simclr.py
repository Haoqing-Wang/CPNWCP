import torch
import torch.nn as nn


def gen_mask(k, feat_dim):
  mask = None
  for i in range(k):
    tmp_mask = torch.triu(torch.randint(0, 2, (feat_dim, feat_dim)), 1)
    tmp_mask = tmp_mask + torch.triu(1 - tmp_mask, 1).t()
    tmp_mask = tmp_mask.view(tmp_mask.shape[0], tmp_mask.shape[1], 1)
    mask = tmp_mask if mask is None else torch.cat([mask, tmp_mask], 2)
  return mask


def entropy(prob):
  # assume m x m x k input
  return -torch.sum(prob * torch.log(prob), 1)


class NT_Xent(nn.Module):
  def __init__(self, batch_size, temperature, mask):
    super(NT_Xent, self).__init__()
    self.batch_size = batch_size
    self.temperature = temperature
    self.mask = mask
    self.criterion = nn.CrossEntropyLoss(reduction="sum")
    self.similarity_f = nn.CosineSimilarity(dim=2)

  def forward(self, z_i, z_j):
    """
    We do not sample negative examples explicitly.
    Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
    """
    p1 = torch.cat((z_i, z_j), dim=0)
    sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

    sim_i_j = torch.diag(sim, self.batch_size)
    sim_j_i = torch.diag(sim, -self.batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.batch_size * 2, 1)
    negative_samples = sim[self.mask].reshape(self.batch_size * 2, -1)

    labels = torch.zeros(self.batch_size * 2).long().cuda()
    logits = torch.cat((positive_samples, negative_samples), dim=1)

    loss = self.criterion(logits, labels)
    loss /= 2 * self.batch_size

    return loss


def mask_correlated_samples(batch_size):
  mask = torch.ones((batch_size*2, batch_size*2)).bool()
  mask = mask.fill_diagonal_(0)
  for i in range(batch_size):
    mask[i, batch_size+i] = 0
    mask[batch_size+i, i] = 0
  return mask


class SimCLR(nn.Module):
  def __init__(self, backbone):
    super(SimCLR, self).__init__()
    self.feature = backbone()
    self.n_features = self.feature.feat_dim
    self.projector = nn.Sequential(nn.Linear(self.n_features, self.n_features//2),
                                   nn.ReLU(),
                                   nn.Linear(self.n_features//2, 128))

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

    mask = mask_correlated_samples(x.size(0))
    criterion = NT_Xent(x.size(0), 0.5, mask)
    loss = criterion(z1, z2)
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