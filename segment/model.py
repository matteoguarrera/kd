import torch
from torch.utils.data import DataLoader
from torchvision import datasets, utils, transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim

from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class MyDataset(Dataset):
    def __init__(self):
        with open('x_seg.npy', 'rb') as f:
            data = np.load(f)
        data = np.swapaxes(data, 1, 3)
        self.data = torch.tensor(np.swapaxes(data, 2, 3))

        with open('y_seg.npy', 'rb') as f:
            self.target = torch.tensor(np.load(f))

        print('Data Loaded Successfully!')

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        return x, y

    def __len__(self):
        return len(self.data)


def train_function(data, model, optimizer, loss_fn, device):
    print('Entering into train function')
    loss_values = []
    for i, (X, y) in enumerate(data):
        if i % 10 == 0:
            print(i, end=' ')
        X, y = X.to(device), y.to(device)
        preds = model(X.float())

        loss = loss_fn(preds, y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

    return loss_values


def validation(model, loader, device):
    # soft = torch.nn.Softmax(dim=1)
    correct_per_batch = []
    for x, y in loader:
        n_tot_pixel_batch = y.reshape(-1).shape[0]

        with torch.no_grad():
            preds = model(x.to(device).float())
            # preds_class = torch.argmax(soft(preds), dim=1)
            preds_class = torch.argmax(preds, dim=1)
            pixelwise_corr = (y == preds_class.cpu())
            # print(preds_class.reshape(-1).shape[0])
            # print(n_tot_pixel_batch)

            correct_per_batch.append(torch.sum(pixelwise_corr) / n_tot_pixel_batch)
        print(f'{np.mean(correct_per_batch):.2f}', end=' ')
    return np.mean(correct_per_batch), preds_class, y


def train_function_distillation(data, model_teacher, model_student, optimizer_student, device):
    print('Entering into train function')
    # soft = torch.nn.Softmax(dim=0)
    loss_fn = nn.MSELoss()

    loss_values = []
    for i, (X, y) in enumerate(data):
        if i % 10 == 0:
            print(i, end=' ')

        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            t_preds = model_teacher(X.float())
            # t_y = torch.argmax(t_preds, dim=1) # retrieve class, no yet logits
        # print(soft(t_preds).shape, y.shape)
        s_preds = model_student(X.float())
        # print(s_preds.shape, t_y.shape)
        loss = loss_fn(s_preds, t_preds)
        optimizer_student.zero_grad()
        loss.backward()
        optimizer_student.step()
        loss_values.append(loss.item())
    return loss_values


# source: https://github.com/hamdaan19/UNet-Multiclass
class UNET(nn.Module):
    
    def __init__(self, layers=[3, 64, 128], classes=1):
        super(UNET, self).__init__()
        self.layers = layers  # 256, 512, 1024
        
        self.double_conv_downs = nn.ModuleList(
            [self.__double_conv__(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])
        
        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
             for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])
            
        self.double_conv_ups = nn.ModuleList(
        [self.__double_conv__(layer, layer//2) for layer in self.layers[::-1][:-2]])
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

    def __double_conv__(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv

    def forward(self, x):
        # down layers
        concat_layers = []
        
        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)
        
        concat_layers = concat_layers[::-1]
        
        # up layers
        for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])
            
            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)
            
        x = self.final_conv(x)
        
        return x 
