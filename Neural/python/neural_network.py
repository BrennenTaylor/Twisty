from __future__ import print_function, division
import os
import math
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from volume_network import *

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
        if i % 100 == 0:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

if __name__ ==  '__main__':
    dtype = torch.float
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(device)

    # Super basic model, not based on any paper
    model = VolumeScatterModel()

    # model = ComplexLuminariesModel()

    model.to(device)

    # Take average of the MSE over the batch
    loss_fn = torch.nn.MSELoss(reduction='mean')

    train_dataset = VolumeScatterDataset(filename='dataset/single_scatter_raymarch/samples.csv', root_dir='dataset/single_scatter_raymarch/')
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=6)

    validation_dataset = VolumeScatterDataset(filename='dataset/single_scatter_raymarch/validation.csv', root_dir='dataset/single_scatter_raymarch/')
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=6)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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

            model_path = 'models/'
            os.makedirs(model_path, exist_ok = True) 
            torch.save(model.state_dict(), os.path.join(model_path, 'model_{}_{}'.format(timestamp, epoch_number)))
            torch.save(model.state_dict(), "latest")
        epoch_number += 1

