import json
from PIL import Image
import h5py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class UnlabelledDataset(Dataset):
    def __init__(self, dataset, datapath, aug_num, backbone):
        self.img_size = (84, 84)
        self.aug_num = aug_num

        # Get the data or paths
        self.dataset = dataset
        self.data = self.extract_data(datapath)

        # Get transform
        self.transform = get_transform(self.img_size, backbone)

    def extract_data(self, datapath):
        # Load miniImagenet
        if self.dataset == 'miniImagenet':
            with h5py.File(os.path.join(datapath, 'train_data.hdf5'), 'r') as f:
                datasets = f['datasets']
                classes = [datasets[k][()] for k in datasets.keys()]
            data = np.concatenate(classes)
        # Load tieredImagenet
        elif self.dataset == 'tieredImagenet':
            data = np.load(os.path.join(datapath, 'train_images.npz'))['images']
        else:
            print("Not support this dataset!")
            assert False

        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        return torch.cat([self.transform(image).unsqueeze(0) for _ in range(self.aug_num)])


def get_transform(img_shape, backbone):
    if backbone == 'Conv4':
        RRC = transforms.RandomResizedCrop(size=img_shape[-1], interpolation=Image.BICUBIC, scale=(0.5, 1.0))
    else:
        RRC = transforms.RandomResizedCrop(size=img_shape[-1], interpolation=Image.BICUBIC)
    data_transforms = transforms.Compose([
        RRC,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return data_transforms