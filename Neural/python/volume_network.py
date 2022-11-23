from __future__ import print_function, division
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class VolumeScatterDataset(Dataset):

    def __init__(self, filename, root_dir):
        self.root_dir = root_dir
        read_data = pd.read_csv(filename, header=None).to_numpy()
        
        # Randomize read data
        print("Randomizing")
        print(read_data.shape)
        np.random.shuffle(read_data)
        print(read_data.shape)

        scaled_input = np.zeros((read_data.shape[0], 9))
        scaled_output = np.zeros((read_data.shape[0], 3))
        
        scaled_input[:, 0] = read_data[:, 0] / 40.
        scaled_input[:, 1] = read_data[:, 1] / 40.
        scaled_input[:, 2] = read_data[:, 2] / 40.

        scaled_input[:, 3] = np.sin(read_data[:, 3]) * np.cos(read_data[:, 4])
        scaled_input[:, 4] = np.sin(read_data[:, 3]) * np.sin(read_data[:, 4])
        scaled_input[:, 5] = np.cos(read_data[:, 3])
 
        scaled_input[:, 6] = read_data[:, 5] / 40.
        scaled_input[:, 7] = read_data[:, 6] / 40.
        scaled_input[:, 8] = read_data[:, 7] / 40.

        scaled_output[:, 0] = read_data[:, 8]
        scaled_output[:, 1] = read_data[:, 9]
        scaled_output[:, 2] = read_data[:, 10]

        self.loaded_data_input = scaled_input
        self.loaded_data_output = scaled_output

    def __len__(self):
        return len(self.loaded_data_input)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_vector = self.loaded_data_input[idx]
        input_vector = np.array([input_vector], dtype=float)

        radiance = self.loaded_data_output[idx]
        radiance = np.array([radiance], dtype=float)

        return input_vector, radiance

class VolumeScatterModel(torch.nn.Module):
    
    def __init__(self):
        super(VolumeScatterModel, self).__init__()

        self.activation = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(9, 1024)
        self.linear2 = torch.nn.Linear(1024, 1024)
        self.linear3 = torch.nn.Linear(1024, 1024)
        self.linear4 = torch.nn.Linear(1024, 1024)
        self.linear5 = torch.nn.Linear(1024, 3)

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

class PositionalEncodingDim3(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, x):

        L = 10
        appended_tensor = torch.empty([x.shape[0], x.shape[1], 6 * L], device=x.device) 
        for level in range(0, L):
            val = 2 ** level

            appended_tensor[:,:,0 + level * 6] = torch.sin(val * torch.pi * x[:,:,0 + self.offset])
            appended_tensor[:,:,1 + level * 6] = torch.cos(val * torch.pi * x[:,:,0 + self.offset])

            appended_tensor[:,:,2 + level * 6] = torch.sin(val * torch.pi * x[:,:,1 + self.offset])
            appended_tensor[:,:,3 + level * 6] = torch.cos(val * torch.pi * x[:,:,1 + self.offset])

            appended_tensor[:,:,4 + level * 6] = torch.sin(val * torch.pi * x[:,:,2 + self.offset])
            appended_tensor[:,:,5 + level * 6] = torch.cos(val * torch.pi * x[:,:,2 + self.offset])

        return appended_tensor

class ResidualBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(512, 512)
        self.activate = torch.nn.LeakyReLU()

    def forward(self, x):
        saved = x
        x = self.linear1(x)
        x += saved
        x = self.activate(x)
        return x

class CachedRadianceModel(torch.nn.Module):
    
    def __init__(self):
        super(CachedRadianceModel, self).__init__()

        self.position_encoding = PositionalEncodingDim3(offset=0)
        self.direction_encoding = PositionalEncodingDim3(offset=3)

        self.linear_input = torch.nn.Linear(60, 256)
        self.activation_input = torch.nn.LeakyReLU()

        self.concat_linear = torch.nn.Linear(512, 512)
        self.concat_linear_activation = torch.nn.LeakyReLU()

        self.residual_unit = ResidualBlock()

        self.output1_linear = torch.nn.Linear(512, 256)
        self.output1_activation = torch.nn.LeakyReLU()

        self.output2_linear = torch.nn.Linear(256, 3)
        self.output2_activation = torch.nn.LeakyReLU()

    def forward(self, x):
        encoded_positions = self.position_encoding(x)
        encoded_directions = self.direction_encoding(x)

        encoded_positions = self.linear_input(encoded_positions)
        encoded_directions = self.linear_input(encoded_directions)

        encoded_positions = self.activation_input(encoded_positions)
        encoded_directions = self.activation_input(encoded_directions)

        concat_input = torch.cat([encoded_positions, encoded_directions], 2)

        saved_concat_input = concat_input

        x = self.concat_linear(concat_input)
        x = self.concat_linear_activation(x)

        x = self.concat_linear(concat_input)
        x = self.concat_linear_activation(x)

        x = self.residual_unit(x)
        x = self.residual_unit(x)
        x = self.residual_unit(x)

        x += saved_concat_input

        x = self.output1_linear(x)
        x = self.output1_activation(x)

        x = self.output2_linear(x)
        x = self.output2_activation(x)

        return x