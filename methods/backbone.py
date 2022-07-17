import math
import torch.nn as nn


# --- gaussian initialize ---
def init_layer(L):
  # Initialization using fan-in
  if isinstance(L, nn.Conv2d):
    n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
    L.weight.data.normal_(0, math.sqrt(2.0/float(n)))
  elif isinstance(L, nn.BatchNorm2d):
    L.weight.data.fill_(1)
    L.bias.data.fill_(0)


# --- flatten tensor ---
class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.reshape(x.size(0), -1)


# --- Simple Conv Block ---
class ConvBlock(nn.Module):
  def __init__(self, indim, outdim, pool=True, padding=1, leaky=False):
    super(ConvBlock, self).__init__()
    self.indim = indim
    self.outdim = outdim
    self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
    self.BN = nn.BatchNorm2d(outdim)
    self.relu = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)

    self.parametrized_layers = [self.C, self.BN, self.relu]
    if pool:
      self.pool = nn.MaxPool2d(2)
      self.parametrized_layers.append(self.pool)

    for layer in self.parametrized_layers:
      init_layer(layer)
    self.trunk = nn.Sequential(*self.parametrized_layers)

  def forward(self, x):
    out = self.trunk(x)
    return out


# --- ConvNet module ---
class ConvNet(nn.Module):
  def __init__(self, depth, flatten=True, leakyrelu=False):
    super(ConvNet, self).__init__()
    trunk = []
    for i in range(depth):
      indim = 3 if i == 0 else 64
      outdim = 64
      B = ConvBlock(indim, outdim, pool=(i < 4), leaky=leakyrelu)
      trunk.append(B)

    if flatten:
      trunk.append(Flatten())
      self.feat_dim = 1600
    else:
      self.feat_dim = [64, 5, 5]
    self.trunk = nn.Sequential(*trunk)

  def forward(self, x):
    out = self.trunk(x)
    return out


# --- Conv networks ---
def Conv4(flatten=True, leakyrelu=True):
    return ConvNet(4, flatten, leakyrelu)


# --- ResNet module ---
class SimpleBlock(nn.Module):
  def __init__(self, indim, outdim, half_res, leaky=False):
    super(SimpleBlock, self).__init__()
    self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
    self.BN1 = nn.BatchNorm2d(outdim)
    self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
    self.BN2 = nn.BatchNorm2d(outdim)
    self.relu1 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
    self.relu2 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
    self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]
    self.half_res = half_res

    # if the input number of channels is not equal to the output, then need a 1x1 convolution
    self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
    self.BNshortcut = nn.BatchNorm2d(outdim)
    self.parametrized_layers.append(self.shortcut)
    self.parametrized_layers.append(self.BNshortcut)

    for layer in self.parametrized_layers:
      init_layer(layer)

  def forward(self, x):
    out = self.C1(x)
    out = self.BN1(out)
    out = self.relu1(out)
    out = self.C2(out)
    out = self.BN2(out)
    short_out = self.BNshortcut(self.shortcut(x))
    out = out + short_out
    out = self.relu2(out)
    return out


class ResNet(nn.Module):
  def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=True, leakyrelu=False):
    super(ResNet, self).__init__()
    assert len(list_of_num_layers) == 4, 'Can have only four stages'
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    bn1 = nn.BatchNorm2d(64)
    relu = nn.ReLU(inplace=True)

    init_layer(conv1)
    init_layer(bn1)
    trunk = [conv1, bn1, relu]

    indim = 64
    for i in range(4):
      for j in range(list_of_num_layers[i]):
        half_res = (j == 0)
        B = block(indim, list_of_out_dims[i], half_res, leaky=leakyrelu)
        trunk.append(B)
        indim = list_of_out_dims[i]

    if flatten:
      trunk.append(nn.AdaptiveAvgPool2d(1))
      trunk.append(Flatten())
      self.feat_dim = 640
    else:
      self.feat_dim = [640, 6, 6]

    self.trunk = nn.Sequential(*trunk)

  def forward(self, x):
    out = self.trunk(x)
    return out


# --- ResNet networks ---
def ResNet10(flatten=True, leakyrelu=True):
  return ResNet(SimpleBlock, [1, 1, 2, 1], [64, 160, 320, 640], flatten, leakyrelu)


def ResNet12(flatten=True, leakyrelu=True):
  return ResNet(SimpleBlock, [1, 1, 2, 2], [64, 160, 320, 640], flatten, leakyrelu)


def ResNet18(flatten=True, leakyrelu=True):
  return ResNet(SimpleBlock, [2, 2, 3, 2], [64, 160, 320, 640], flatten, leakyrelu)


model_dict = dict(Conv4=Conv4, ResNet10=ResNet10, ResNet12=ResNet12, ResNet18=ResNet18)