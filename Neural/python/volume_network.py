from __future__ import print_function, division
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class VolumeScatterDataset(Dataset):

    def __init__(self, filename, root_dir):
        self.root_dir = root_dir
        read_data = pd.read_csv(filename, header=None).to_numpy()

        scaled_data = np.zeros((read_data.shape[0], 12))
        
        scaled_data[:, 0] = read_data[:, 0] / 10.
        scaled_data[:, 1] = read_data[:, 1] / 10.
        scaled_data[:, 2] = read_data[:, 2] / 10.

        scaled_data[:, 3] = np.sin(read_data[:, 3]) * np.cos(read_data[:, 4])
        scaled_data[:, 4] = np.sin(read_data[:, 3]) * np.sin(read_data[:, 4])
        scaled_data[:, 5] = np.cos(read_data[:, 3])
 
        scaled_data[:, 6] = read_data[:, 5] / 40.
        scaled_data[:, 7] = read_data[:, 6] / 40.
        scaled_data[:, 8] = read_data[:, 7] / 40.

        scaled_data[:, 9] = read_data[:, 8]
        scaled_data[:, 10] = read_data[:, 9]
        scaled_data[:, 11] = read_data[:, 10]

        self.loaded_data = scaled_data

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_vector = self.loaded_data[idx][0:9]
        input_vector = np.array([input_vector], dtype=float)

        radiance = self.loaded_data[idx,][9:12]
        radiance = np.array([radiance], dtype=float)

        return input_vector, radiance

class VolumeScatterModel(torch.nn.Module):
    
    def __init__(self):
        super(VolumeScatterModel, self).__init__()

        self.activation = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(9, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 512)
        self.linear4 = torch.nn.Linear(512, 512)
        self.linear5 = torch.nn.Linear(512, 3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.activation(x)
        x = self.linear5(x)
        x = self.activation(x)
        return x