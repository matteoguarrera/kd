import torch
from torch.utils.data import DataLoader
from torchvision import datasets, utils, transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class MyDataset(Dataset):
    def __init__(self):
        with open('../x_seg.npy', 'rb') as f:
            data = np.load(f)
        data = np.swapaxes(data, 1, 3)
        self.data = torch.tensor(np.swapaxes(data, 2, 3))

        with open('../y_seg.npy', 'rb') as f:
            self.target = torch.tensor(np.load(f))

        print('Data Loaded Successfully!')

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        return x, y

    def __len__(self):
        return len(self.data)