import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class FCNet(nn.Module):
    '''全连接网络'''
    def __init__(self):
        super(FCNet,self).__init__()
        self.fc_model = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

        
    def forward(self, input: torch.TensorType):
        # print('input: ', input.shape)
        '''由于是灰度图，最开始的输入input尺寸为(batch_size, 1, 28, 28)'''
        x = torch.flatten(input, 1) # x: (batch_size, 28*28)
        output = self.fc_model(x)
        return output