import os
import h5py
import json
import torch
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from abc import abstractmethod


def load_data(file):
  try:
    with open(file, 'rb') as fo:
      data = pickle.load(fo)
    return data
  except:
    with open(file, 'rb') as f:
      u = pickle._Unpickler(f)
      u.encoding = 'latin1'
      data = u.load()
    return data


class SubDataset:
  def __init__(self, sub_meta, cl, transform):
    self.sub_meta = sub_meta
    self.cl = cl
    self.transform = transform

  def __getitem__(self, i):
    img = Image.fromarray(self.sub_meta[i])
    img = self.transform(img)
    target = self.cl
    return img, target

  def __len__(self):
    return self.sub_meta.shape[0]


class SetDataset:
  def __init__(self, dataset, data_file, num_image, transform, split):
    self.dataset = dataset
    self.sub_meta = self.extract_data(data_file, split)
    self.sub_dataloader = []
    for cl in range(len(self.sub_meta)):
      sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
      self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, batch_size=num_image, shuffle=True))

  def extract_data(self, data_file, split):
    # Load miniImagenet
    if self.dataset == 'miniImagenet':
      with h5py.File(os.path.join(data_file, '%s_data.hdf5'%split), 'r') as f:
        datasets = f['datasets']
        classes = [datasets[k][()] for k in datasets.keys()]
    # Load tieredImagenet
    elif self.dataset == 'tieredImagenet':
      labels = load_data(os.path.join(data_file, '%s_labels.pkl'%split))['labels']
      dataset = np.load(os.path.join(data_file, '%s_images.npz')%split)['images']
      s, classes = 0, []
      for e in range(1, len(labels)):
        if labels[e]!=labels[e-1]:
          classes.append(dataset[s:e])
          s = e
      classes.append(dataset[s:])
    else:
        print("Not support this dataset!")
        assert False

    return classes

  def __getitem__(self, i):
    return next(iter(self.sub_dataloader[i]))

  def __len__(self):
    return len(self.sub_meta)


class EpisodicBatchSampler(object):
  def __init__(self, n_classes, n_way, n_episodes):
    self.n_classes = n_classes
    self.n_way = n_way
    self.n_episodes = n_episodes

  def __len__(self):
    return self.n_episodes

  def __iter__(self):
    for i in range(self.n_episodes):
      yield torch.randperm(self.n_classes)[:self.n_way]


class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file, split):
    pass


class SetDataManager(DataManager):
  def __init__(self, dataset, n_way, n_support, n_query, n_eposide=100):
    super(SetDataManager, self).__init__()
    self.dataset = dataset
    self.n_way = n_way
    self.num_image = n_support + n_query
    self.n_eposide = n_eposide
    self.img_size = (84, 84)
    self.transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

  def get_data_loader(self, data_file, split):
    dataset = SetDataset(self.dataset, data_file, self.num_image, self.transform, split)
    sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=4)
    return data_loader