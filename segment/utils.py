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


# import torchvision.transforms.functional as TF


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

