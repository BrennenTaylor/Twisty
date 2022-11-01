from __future__ import print_function, division
import os
import math
import torch
import torchvision
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class VolumeScatterDataset(Dataset):

    def __init__(self, filename, root_dir):
        self.root_dir = root_dir
        read_data = pd.read_csv(filename, header=None).to_numpy()


        scaled_data = np.zeros((read_data.shape[0], 12))
        
        scaled_data[:, 0] = read_data[:, 0] / 10
        scaled_data[:, 1] = read_data[:, 1] / 10
        scaled_data[:, 2] = read_data[:, 2] / 10

        scaled_data[:, 3] = np.sin(read_data[:, 3]) * np.cos(read_data[:, 4])
        scaled_data[:, 4] = np.sin(read_data[:, 3]) * np.sin(read_data[:, 4])
        scaled_data[:, 5] = np.cos(read_data[:, 3])
 
        scaled_data[:, 6] = read_data[:, 5] / 20
        scaled_data[:, 7] = read_data[:, 6] / 20
        scaled_data[:, 8] = read_data[:, 7] / 20

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


dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print(device)

model = VolumeScatterModel()
model.to(device)

loss_fn = torch.nn.MSELoss(reduction='mean')


train_dataset = VolumeScatterDataset(filename='dataset/single_scatter_raymarch/samples.csv', root_dir='dataset/single_scatter_raymarch/')
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, pin_memory=True)

validation_dataset = VolumeScatterDataset(filename='dataset/single_scatter_raymarch/validation.csv', root_dir='dataset/single_scatter_raymarch/')
validation_dataloader = DataLoader(validation_dataset, batch_size=1024, shuffle=True, pin_memory=True)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_dataloader):
        
        train_input, train_output = data
        
        train_input = train_input.float().to(device)
        train_output = train_output.float().to(device)


        # import pdb; pdb.set_trace()
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        model_prediction = model(train_input)

        # Compute the loss and its gradients
        loss = loss_fn(model_prediction, train_output)
        # print(loss)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 0:
            last_loss = running_loss / 10 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/volume_scatter_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_dataloader):
        vinputs, vGT = vdata

        vinputs = vinputs.float().to(device)
        vGT = vGT.float().to(device)

        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vGT)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

        
        # torch.onnx.export(model, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

    epoch_number += 1