import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import copy
from skimage import io, transform
from parse import *
import pandas as pd
from numpy import save, load

from torch.utils.data import Dataset, DataLoader


def create_dataset_file():
    LOAD = True

    # Top level data directory.
    data_dir = "./imgs/follower/img/"

    if not LOAD:
        # /home/matteogu/Documents/kd/imgs/follower/img/
        lst_files = os.listdir(data_dir)
        print(len(lst_files))

        # Filter Data
        data = np.zeros((len(lst_files), 3))
        name = [None] * len(lst_files)

        labelled = 0
        for ii, file in enumerate(lst_files):
            results = parse("{}_{}_{}_{}_{}_{}.jpg", file)
            if results is None:
                results = parse("{}_{}_{}_{}_{}.jpg", file)  # orientation_to_lead
                idx, orientation_to_lead = results[-2:]
                # print(file, orientation_to_lead)
            else:
                idx, orientation_to_lead, distance_to_lead = results[-3:]
                # print(file, orientation_to_lead, distance_to_lead)
                data[labelled] = idx, orientation_to_lead, distance_to_lead
                name[labelled] = file
                labelled += 1

        df = pd.DataFrame([name[:labelled], data[:labelled, 0], data[:labelled, 1], data[:labelled, 2]]).T
        df.columns = ['filename', 'idx', 'orientation', 'distance']
        df_follower_labelled = df[df['orientation'] != 99999.99]
        # df_follower_labelled.to_csv('df_follower_labelled.csv')
        df_follower_labelled.reset_index(drop=True, inplace=True)
        # df_follower_labelled
        # Read Data
        dataset = np.zeros((len(df_follower_labelled), 64, 128, 3), dtype=np.uint8)
        for i, row in df_follower_labelled.iterrows():
            image = io.imread(data_dir + row['filename'])
            dataset[i] = image

        # Save data to npy file
        save('df_follower_labelled.npy', dataset)
    else:
        dataset_loaded = load('df_follower_labelled.npy')
        df_loaded = pd.read_csv('df_follower_labelled.csv')


class WeBotsDataset(Dataset):
    def __init__(self, csv_file='df_follower_labelled.csv',
                 root_dir='./imgs/follower/img/', train=True,
                 transform=None):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.df_loaded = pd.read_csv(csv_file)
        self.dataset_loaded = torch.Tensor(load('df_follower_labelled.npy')/255)  # .to(self.device) # to float

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df_loaded)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.dataset_loaded[idx]
        target = torch.tensor(self.df_loaded.iloc[idx]['orientation'])  # .to(self.device)
        # sample = {'image': image, 'landmarks': landmarks}

        # if self.transform:
        #     sample = self.transform(image)

        return image, target
