import torch
import torch.nn as nn


# Creating a CNN class
class ConvNeuralNet(nn.Module):
	#  Determine what layers and their order in CNN object
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        # self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3)
        # self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv_layer3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3)
        # self.conv_layer4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # self.fc1 = nn.Linear(61000, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(7440, num_classes)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
#         print(out.shape)
#         out = self.conv_layer2(out)
#         print(out.shape)
#         out = self.max_pool1(out)
#         print(out.shape)
        out = self.conv_layer3(out)
#         print(out.shape)
#         out = self.conv_layer4(out)
#         print(out.shape)
        out = self.max_pool2(out)
#         print(out.shape)
        out = out.reshape(out.size(0), -1)
#         print(out.shape)
#         out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out.squeeze()
