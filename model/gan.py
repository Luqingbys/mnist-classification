import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



class Generator(nn.Module):
    def __init__(self, nz, ngf, num_classes, nc=1):
        super(Generator, self).__init__()
        self.image_path = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True)
        )
        self.label_path = nn.Sequential(
            nn.ConvTranspose2d(num_classes, ngf*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True)
        )
        self.main_path = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, image, label):
        image = self.image_path(image)
        label = self.label_path(label)
        inp = torch.cat((image, label), dim=1)
        return self.main_path(inp)


class Discriminator(nn.Module):
    def __init__(self, ndf, num_classes=10, nc=1):
        super(Discriminator, self).__init__()
        self.image_path = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.label_path = nn.Sequential(
            nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.main_path = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image, label):
        image = self.image_path(image)
        label = self.label_path(label)
        inp = torch.cat((image, label), dim=1)
        return self.main_path(inp)