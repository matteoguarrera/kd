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


def create_dataset_file(load=True, filename='df_follower_labelled'):

    # Top level data directory.
    data_dir = "./imgs/follower/img/"

    if not load:
        # /home/matteogu/Documents/kd/imgs/follower/img/
        lst_files = os.listdir(data_dir)
        print(len(lst_files))

        # Filter Data
        data = np.zeros((len(lst_files), 4))
        name = [None] * len(lst_files)

        labelled = 0
        for ii, file in enumerate(lst_files):
            results = parse("{}_{}_{}_{}_{}_{}_{}.jpg", file)
            if results is None:
                results = parse("{}_{}_{}_{}_{}.jpg", file)  # orientation_to_lead
                idx, orientation_to_lead = results[-2:]
                # print(file, orientation_to_lead)
            else:
                idx, orientation_to_lead, distance_to_lead, filtered_angle = results[-4:]
                # print(file, orientation_to_lead, distance_to_lead)
                data[labelled] = idx, orientation_to_lead, distance_to_lead, filtered_angle
                name[labelled] = file
                labelled += 1

        df = pd.DataFrame([name[:labelled], data[:labelled, 0], data[:labelled, 1],
                                            data[:labelled, 2], data[:labelled, 3]]).T

        df.columns = ['filename', 'idx', 'orientation', 'distance', 'angle']
        df_follower_labelled = df[df['orientation'] != 99999.99]
        df_follower_labelled.reset_index(drop=True, inplace=True)
        # df_follower_labelled
        # Read Data
        dataset = np.zeros((len(df_follower_labelled), 64, 128, 3), dtype=np.uint8)
        for i, row in df_follower_labelled.iterrows():
            image = io.imread(data_dir + row['filename'])
            dataset[i] = image

        # Save data to npy file
        save(f'{filename}.npy', dataset)
        df_follower_labelled.to_csv(f'{filename}.csv')

    else:
        dataset_loaded = load(f'{filename}.npy')
        df_loaded = pd.read_csv(f'{filename}.csv')
        return dataset_loaded, df_loaded


class WeBotsDataset(Dataset):
    def __init__(self, filename='df_follower_labelled2',
                 # root_dir='./imgs/follower/img/',
                 transform=None
                 ):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.df_loaded = pd.read_csv(f'{filename}.csv')
        dataset_loaded = load(f'{filename}.npy')/255.0  # .to(self.device) # to float
        self.dataset_loaded = dataset_loaded.astype(np.float32)
        # self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df_loaded)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.dataset_loaded[idx]
        orientation = torch.tensor(self.df_loaded.iloc[idx]['orientation']).float()  # .to(self.device)
        angle = torch.tensor(self.df_loaded.iloc[idx]['angle']).float()  # .to(self.device)

        # sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            image = self.transform(image)

        return image, (orientation, angle)
