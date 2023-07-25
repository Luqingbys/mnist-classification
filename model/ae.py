import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import datetime


class mAutoEncoder(nn.Module):
    """卷积自编码器"""
    def __init__(self, inp_dim=1):
        super(mAutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inp_dim, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), padding=1)

    def encode(self, x):
        print('encode: ', self.conv2(self.downsample(self.conv1(x))).shape)
        return self.conv2(self.downsample(self.conv1(x)))

    def decode(self, x):
        return self.conv4(self.upsample(self.conv3(x)))

    def forward(self, x):
        return self.decode(self.encode(x))


# class mConvVAE(nn.Module):
#     def __init__(self, latent_dims=4):
#         super(ConvVAE, self).__init__()
#         self.encoder = ConvEncoder(latent_dims)
#         self.decoder = ConvDecoder(latent_dims)

#     def forward(self, x):
#         # x = x.to(device)
#         # print('in ConvVAE: ', x.shape)
#         z = self.encoder(x)
#         return self.decoder(z)
 

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


class Unflatten(nn.Module):
	def __init__(self, channel, height, width):
		super(Unflatten, self).__init__()
		self.channel = channel
		self.height = height
		self.width = width

	def forward(self, input):
		return input.view(input.size(0), self.channel, self.height, self.width)
    
 
class AE(nn.Module):
    def __init__(self, input_channels=1, z_dim=20) -> None:
        super(AE, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
			nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
			nn.ReLU(), 
			Flatten(),
			nn.Linear(6272, 1024),
			nn.ReLU(),
            nn.Linear(1024, z_dim),
            nn.ReLU()
		) # (b, 1, 28, 28) => (b, 1024) => (b, 20)
        self.decoder = nn.Sequential(
			nn.Linear(self.z_dim, 1024),
			nn.ReLU(),
			nn.Linear(1024, 6272),
			nn.ReLU(),
			Unflatten(128, 7, 7),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
			nn.Sigmoid()
		)
    
    def encode(self, x):
        h = self.encoder(x)
        # print('h: ', h.shape)
        return h
    
    def decode(self, h):
        return self.decoder(h)

    def forward(self, x):
        return self.decode(self.encode(x))


# if __name__ == '__main__':
#     net = AE(z_dim=20)
#     inp = torch.rand(8, 1, 28, 28)
#     res = net(inp)
#     print('res:', res.shape)