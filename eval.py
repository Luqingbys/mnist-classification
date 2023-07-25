import torch
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.dataloader import sampleDataset
from encoders import *
import argparse
from model.classifier import Classifier, Mlp
import os
import time


parser = argparse.ArgumentParser('MNIST classify')
parser.add_argument('--classifier', default='knn', type=str)
parser.add_argument('--batch', default=32, type=int)
parser.add_argument('--device', default='cuda: 0', type=str)
parser.add_argument('--ratio', default=0.1, type=float)
parser.add_argument('--model', default='Vae', type=str)
parser.add_argument('--epochs', default=50, type=int, help='epochs to train classifier.')
args = parser.parse_args()


#这个函数包括了两个操作：将图片转换为张量，以及将图片进行归一化处理
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

ratio = args.ratio
batch_size = args.batch
train_df = pd.read_csv(f'./csv/train_{ratio}_classify.csv')
train_ds = sampleDataset(train_df, transforms=transform)

# transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                                 torchvision.transforms.Normalize(mean = [0.5],std = [0.5])])
trainDataLoader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_ds = torchvision.datasets.MNIST('./data', train=False, transform = transform)
testDataLoader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# processBar = tqdm(trainDataLoader, unit='step')

if args.device == 'cpu':
    device = 'cpu'
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = torch.load('output/Conv_VAE/0.1/2023-04-04-1680609788.0/conv_vae.pth', map_location=device)
model = Ae(model_path="/home/xiao/mywork/mnist-classify/output/AE/0.1/2023-04-10-1681101709.0/weight/model_final.pth", device=device)


X = []
y = []
X_test = []
y_test = []
for imgs, labels in trainDataLoader:
    # print(imgs.shape)
    imgs = imgs.to(device)
    print(imgs.shape)
    # imgs = imgs.flatten(1) # (batch_size, 1, 28, 28) => (batch_size, 28*28)
    labels = labels.to(device)
    # mu, logvar = model.encode(imgs)
    # feature = model.reparametrize(mu, logvar).cpu().detach().numpy() # (batch_size, 20)
    feature = model.encode(imgs)
    # print('feature: ', type(feature), feature.shape)
    X.append(feature) # (batch_size, 20)
    y.append(labels.cpu().detach().numpy()) # (batch_size,)


for imgs, labels in testDataLoader:
    imgs = imgs.to(device)
    # imgs = imgs.flatten(1)
    # labels = labels
    # mu, logvar = model.encode(imgs)
    # feature = model.reparametrize(mu, logvar).cpu().detach().numpy()
    feature = model.encode(imgs)
    X_test.append(feature)
    y_test.append(labels)


# print(y_test.shape)
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)
X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)
# X = np.asarray(X) # (188, )
# y = np.asarray(y)

print(X.shape, y.shape, X_test.shape, y_test.shape)

predict = None

classifier_model = 'knn' if args.classifier == 'knn' else 'Mlp'
pre_model = 'Vae' if args.model == 'Vae' else 'Ae'
output_path = f'output/classifier/{classifier_model}/' + str(args.ratio) + '/' + pre_model + '/' + time.strftime('%Y-%m-%d-', time.localtime()) + str(time.mktime(time.localtime())) + '/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

if args.classifier == 'knn':
    classifier = KNeighborsClassifier(n_neighbors=5)
    # classifier = AdaBoostClassifier()
    # model = Mlp()
    # classifier = Classifier(model=model, features=X, labels=y, test_features=X_test, test_labels=y_test, lossFunc=loss, optimizer=optimizer, n_epoch=50, device=device, batch_size=32)
    classifier.fit(X, y)
    predict = classifier.predict(X_test)
else:
    model = Mlp(latent_dim=20)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    classifier = Classifier(model=model, features=X, labels=y, test_features=X_test, test_labels=y_test, lossFunc=loss, optimizer=optimizer, n_epoch=args.epochs, device=device, batch_size=batch_size)
    classifier.fit(output=output_path)
    predict = classifier.predict()

print("accuracy_score: %.4lf" % accuracy_score(predict, y_test))
report = classification_report(y_test, predict, output_dict=True)
print("Classification report for classifier %s:\n%s\n" % (classifier, classification_report(y_test, predict)))

save_df = pd.DataFrame(report)
save_df.to_csv(output_path + f'epochs-{args.epochs}_result.csv', index=True)
