from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class VolumeScatterDataset(Dataset):

    def __init__(self, filename, root_dir, transform=None):
        self.loaded_data = pd.read_csv(filename, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        viewPos = self.loaded_data.iloc[idx, 0:2]
        viewPos = np.array([viewPos])

        viewDir = self.loaded_data.iloc[idx, 3:4]
        viewDir = np.array([viewDir])

        lightPos = self.loaded_data.iloc[idx, 5:7]
        lightPos = np.array([lightPos])

        radiance = self.loaded_data.iloc[idx, 8:10]
        radiance = np.array([radiance])

        sample = {
            'viewPos': viewPos,
            'viewDir': viewDir,
            'lightPos': lightPos,
            'radiance': radiance
        }

        if self.transform:
            sample = self.transform(sample)
        
        return sample

single_scatter_raymarch_dataset = VolumeScatterDataset(filename='dataset/single_scatter_raymarch/samples_10_6.csv', root_dir='dataset/single_scatter_raymarch/')

print(len(single_scatter_raymarch_dataset))

# for i in range(len(single_scatter_raymarch_dataset)):
#     sample = single_scatter_raymarch_dataset[i]
#     print(i, sample['viewPos'].shape, sample['viewDir'].shape, sample['lightPos'].shape, sample['radiance'].shape)