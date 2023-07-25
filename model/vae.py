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


class VAE(nn.Module):
    '''
    变分自编码器
    '''

    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 100)
        self.fc22 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        '''编码器，得到的结果是高斯分布的均值和协方差'''
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        '''重参数化，因为采样过程不能微分'''
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        '''解码器'''
        h3 = F.relu(self.fc3(z))
        # return F.sigmoid(self.fc4(h3))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # print('x: ', x.shape)
        # x: (batch_size, 28*28) => mu, logvar: (batch_size, 20)
        mu, logvar = self.encode(x)
        # print('after encoder, mu and logver: ', mu.shape, logvar.shape)
        z = self.reparametrize(mu, logvar)  # z: (batch_size, 20)
        # print('after reparametrize: ', z.shape)
        return z, self.decode(z), mu, logvar  # output: (batch_size, 28*28)


class ConvEncoder(nn.Module):
    '''
    卷积变分自编码器
    '''

    def __init__(self, latent_dims) -> None:
        super().__init__()  # input: (batch_size, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(3*3*32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0


    def forward(self, x):
        # x = x.to(device)
        x = F.relu(self.conv1(x))
        # print('before bn2 x:', x.shape)
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class ConvDecoder(nn.Module):
    def __init__(self, latent_dims) -> None:
        super().__init__()
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class mConvVAE(nn.Module):
    def __init__(self, latent_dims=4):
        super(ConvVAE, self).__init__()
        self.encoder = ConvEncoder(latent_dims)
        self.decoder = ConvDecoder(latent_dims)

    def forward(self, x):
        # x = x.to(device)
        # print('in ConvVAE: ', x.shape)
        z = self.encoder(x)
        return self.decoder(z)
 

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


class ConvVAE(nn.Module):
	def __init__(self, input_channels=1, z_dim=20):
		super(ConvVAE, self).__init__()
		self.z_dim = z_dim
		self.encoder = nn.Sequential(
			nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			Flatten(),
			nn.Linear(6272, 1024),
			nn.ReLU()
		) # (b, 1, 28, 28) => (b, 1024)

		# hidden => mu
		self.fc1 = nn.Linear(1024, self.z_dim)
		# hidden => logvar
		self.fc2 = nn.Linear(1024, self.z_dim)

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
		mu, logvar = self.fc1(h), self.fc2(h)
		return mu, logvar

	def decode(self, z):
		z = self.decoder(z)
		return z

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar
