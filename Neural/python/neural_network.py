from __future__ import print_function, division
import os
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

    def __init__(self, filename, root_dir, transform=None):
        self.loaded_data = pd.read_csv(filename, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_vector = self.loaded_data.iloc[idx, 0:8]
        input_vector = np.array([input_vector], dtype=float)

        radiance = self.loaded_data.iloc[idx, 8:11]
        radiance = np.array([radiance], dtype=float)

        if self.transform:
            input_vector = self.transform(input_vector)

        if self.transform:
            radiance = self.transform(radiance)

        return input_vector, radiance

class VolumeScatterModel(torch.nn.Module):
    
    def __init__(self):
        super(VolumeScatterModel, self).__init__()

        self.linear1 = torch.nn.Linear(8, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 3)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


dtype = torch.float
device = torch.device("cpu")

model = VolumeScatterModel()

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-6


train_dataset = VolumeScatterDataset(filename='dataset/single_scatter_raymarch/samples.csv', root_dir='dataset/single_scatter_raymarch/')
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

validation_dataset = VolumeScatterDataset(filename='dataset/single_scatter_raymarch/validation.csv', root_dir='dataset/single_scatter_raymarch/')
validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True)


optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_dataloader):
        
        train_input, train_output = data
        
        train_input = train_input.float()
        train_output = train_output.float()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        model_prediction = model(train_input)

        # Compute the loss and its gradients
        loss = loss_fn(model_prediction, train_output)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
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
    for i, vdata in enumerate(validation_loader):
        vinputs, vGT = vdata

        vinputs = vinputs.float()
        vGT = vGT.float()

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

    epoch_number += 1