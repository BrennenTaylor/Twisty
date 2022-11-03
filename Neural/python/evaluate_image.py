from __future__ import print_function, division
import os
import math
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import cv2

from volume_network import *

dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print(f"Supported device: {device}")

model = VolumeScatterModel()
model.load_state_dict(torch.load("latest"))
model = model.to(device)

read_data = pd.read_csv("dataset\single_scatter_raymarch\image_samples.csv", header=None).to_numpy()
print(f"Read data: {read_data[0]}")

pixel_mask = np.zeros((read_data.shape[0], 1))
pixel_mask = np.copy(read_data[:, 0])
print(f"Pixel mask: {pixel_mask[0]}")

scaled_data = np.zeros((read_data.shape[0], 6))
scaled_data[:, 0] = read_data[:, 1] / 10
scaled_data[:, 1] = read_data[:, 2] / 10.
scaled_data[:, 2] = read_data[:, 3] / 10.

scaled_data[:, 3] = read_data[:, 4]
scaled_data[:, 4] = read_data[:, 5]
scaled_data[:, 5] = read_data[:, 6]

# scaled_data[:, 6] = 0.
# scaled_data[:, 7] = 40.
# scaled_data[:, 8] = 10.

print(f"Scaled Data: {scaled_data[0]}")

pixel_inputs = torch.from_numpy(scaled_data).float()
pixel_inputs = pixel_inputs.to(device)

pixel_outputs = model(pixel_inputs)
pixel_outputs = pixel_outputs.to(device='cpu')

pixel_outputs = pixel_outputs.detach().numpy()
print(f"Pixel output: {pixel_outputs[0]}")

img_width = int(math.sqrt(pixel_outputs.shape[0]))
pixel_outputs = pixel_outputs.reshape((img_width, img_width, 3))

pixel_mask = pixel_mask.reshape(img_width, img_width, 1)
pixel_mask_img = np.zeros((img_width, img_width, 3))
pixel_mask_img[:, :, 0] = np.copy(pixel_mask[:, :, 0])
pixel_mask_img[:, :, 1] = np.copy(pixel_mask[:, :, 0])
pixel_mask_img[:, :, 2] = np.copy(pixel_mask[:, :, 0])

pixel_mask_img = pixel_mask_img.astype(np.float32)

pixel_outputs = np.multiply(pixel_outputs, pixel_mask_img)
pixel_outputs = pixel_outputs * 25.
print(f"Reshaped pixel: {pixel_outputs[0][0]}")

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

cv2.imwrite('network_image.exr', pixel_outputs)
cv2.imwrite('pixel_mask.exr', pixel_mask_img)