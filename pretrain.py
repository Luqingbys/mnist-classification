import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.resNet import ResNet, BasicConv
from model.vae import ConvVAE
from utils.logger import get_logger
from utils.dataloader import sampleDataset
import argparse
import importlib
import os
import time
from trainConvVAE import train
from trainAE import train_ae


parser = argparse.ArgumentParser(description='mnist手写数字识别数据集图像分类程序，预训练')
parser.add_argument('--module', type=str, help='select a module file to do classification', default='vae')
parser.add_argument('--model', type=str, help='select a model to do classification', default='ConvVAE')
parser.add_argument('--output', type=str, help='select a path to save the running result', default='output')
parser.add_argument('--ratio', type=float, default=0.1, help='input a ratio to sample from MNSIT.')
parser.add_argument('--use_variant', type=bool, default=True, help='wheather to use variant auto encoder or not.')
parser.add_argument('--use_shift', type=bool, default=False, help='wheather to use circle shift to the datasets or not.')
args = parser.parse_args()


# 如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = 'cpu'
path = './data/'  # 数据集下载后保存的目录
# 设定每一个Batch的大小
BATCH_SIZE = 16

# 这个函数包括了两个操作：将图片转换为张量，以及将图片进行归一化处理
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                                             torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])])

# 训练集不全选时执行，读取训练集.csv文件
# train_df = pd.read_csv(f'./train_{1-args.ratio}_rest.csv')
train_df = pd.read_csv(f'./csv/train_{args.ratio}_semi.csv')
valid_df = pd.read_csv(f'./csv/valid_{args.ratio}_semi.csv')
train_ds = sampleDataset(df=train_df, translation=args.use_shift)
valid_ds = sampleDataset(df=valid_df, translation=args.use_shift)
# 构建训练集、验证集和测试集的DataLoader
trainDataLoader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
validDataLoader = torch.utils.data.DataLoader(dataset=valid_ds, batch_size=BATCH_SIZE, shuffle=False)
# 下载训练集和测试集
testDataset = torchvision.datasets.MNIST(path, train=False, transform=transform)
testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=32, shuffle=False)


# 动态定义好模型，优化器，损失函数
# net = importlib.import_module('.'+args.module+'.'+args.model, package='model')
module = __import__('model.' + args.module, fromlist=[''])
model = getattr(module, args.model)
net: nn.Module = model()
# print(net)
net = net.to(device)

# loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

output_path = args.output + '/' + args.model + '/' + str(args.ratio) + '/' + time.strftime('%Y-%m-%d-',
                                                                                           time.localtime()) + str(
    time.mktime(time.localtime())) + '/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

if args.use_variant:
    train(net=net, z_dim=20, batch_size=BATCH_SIZE, trainloader=trainDataLoader, validloader=validDataLoader, epochs=50, valid_every=5, output=output_path, device=device)
else:
    train_ae(net=net, z_dim=20, batch_size=BATCH_SIZE, trainloader=trainDataLoader, validloader=validDataLoader, epochs=50, valid_every=5, output=output_path, device=device)