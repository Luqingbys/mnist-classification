import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from resNet import ResNet
import numpy as np
import sys
from tqdm import tqdm
sys.path.append('..')
from utils.logger import get_logger


class FeatureData(Dataset):
    '''
    读取提取到的特征数据，做成新的数据集，喂给分类器
    features: np.ndarray, (5999, 4)
    labels: np.ndarray, (5999,) 
    '''
    def __init__(self, features, labels) -> None:
        super().__init__()
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return torch.Tensor(self.features[index]), int(self.labels[index])
    
    def __len__(self):
        return self.features.shape[0]


class Mlp(nn.Module):
    '''全连接网络'''
    def __init__(self, latent_dim=4, hidden_dim=48, num_class=10):
        super(Mlp, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_class)
        )
        
    def forward(self, x: torch.TensorType):
        # print('input: ', input.shape)
        '''x: (batch_size, latent_dim=4)'''
        # x = torch.flatten(input, 1) # x: (batch_size, 28*28)
        output = self.mlp(x)
        return output


class ResNetModel(nn.Module):
    def __int__(self, latent_dim):
        super(ResNetModel, self).__int__()
        self.conv1 = nn.Conv2d(1024, )
        self.resNet = ResNet()
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor):
        """x: (batch_size, latent_dim)"""
        x = x.reshape(x.shape[0], 1, self.latent_dim // 2, -1)  # => (batch_size, 1, latent_dim//2, latent_dim//2)
        return self.resNet(x)


class Classifier():
    '''
    将分类器连同训练、预测全部封装到一个类中
    model: 要训练的分类器，由于是多个模型套在一起，这里将分类器包含在一个类中
    '''
    
    def __init__(self, model: nn.Module, features, labels, test_features, test_labels, lossFunc, optimizer, n_epoch, device="cpu", batch_size=32):
        self.model = model.to(device)  # 它才是要训练的对象
        self.device = device
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.train_dataloader, self.test_dataloader = self.load_data(features, labels, test_features, test_labels)
        self.loss_f = lossFunc
        self.optimizer = optimizer

    
    def load_data(self, features, labels, test_features, test_labels, batch_size):
        #构建数据集和测试集的DataLoader
        trainData = FeatureData(features=features, labels=labels)
        testData = FeatureData(features=test_features, labels=test_labels)
        trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True)
        testDataLoader = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True)
        return trainDataLoader, testDataLoader
        

    def predict(self):
        '''使用该类的model属性进行预测，它实际上是训练完成后保存好的模型'''
        self.model.eval()

        # all_pred_prob = []
        # with torch.no_grad():
        #     for batch_idx, batch in enumerate(dataloader):
        #         input_x, input_y = tuple(t.to(self.device) for t in batch)
        #         pred = self.model(input_x)
        #         all_pred_prob.append(pred.cpu().data.numpy())
        
        # all_pred_prob = np.concatenate(all_pred_prob)
        # all_pred = np.argmax(all_pred_prob, axis=1)
        # return all_pred

        # net = torch.load(output+'model.pkl')
        all_labels = []
        all_predicts = []
        all_predict_num = [] # 对测试集的预测结果，0~9
        eye = np.eye(10)

        # 构造临时变量
        correct, totalLoss = 0, 0
        # 关闭模型的训练状态
        self.model.train(False)
        with torch.no_grad():
            # 对测试集的DataLoader进行迭代
            processBar = tqdm(self.test_dataloader, unit='step')
            for step, (testImgs, labels) in enumerate(processBar):
                # print('测试数据集: ', testImgs.shape)
                testImgs = testImgs.to(self.device)
                labels = labels.to(self.device)
                # print(testImgs.shape)
                # print(labels)
                outputs: torch.Tensor = self.model(testImgs)
                # print(outputs)
                loss = self.loss_f(outputs, labels)

                all_predicts.append(outputs.to('cpu').detach().numpy()) # 直接添加当前批量的预测结果，outputs: (batch_size, 10)
                all_labels.append(eye[labels.to('cpu')]) # 通过单位阵eye得到labels的全部独热编码

                # predictions: (256, )，刚好是当前批量每一个样本的预测结果
                predictions = torch.argmax(outputs, dim=1)
                all_predict_num.append(predictions.cpu().numpy()) # predictions在cuda上时，先将其转移到cpu
                
                # 存储测试结果
                totalLoss += loss
                cur_correct = torch.sum(predictions == labels)
                correct += cur_correct
                # 将本step结果进行可视化处理
                processBar.set_description("current batch testing result... Test Loss: %.4f, Test Acc: %.4f" % (loss.item(), cur_correct / self.batch_size))
        all_predict_num = np.concatenate(all_predict_num)
        return all_predict_num
    

    def fit(self, output):
        '''
        训练，得到模型最优参数
        X: (sample, latent_dim=4)
        '''
        logger = get_logger(path=output, filename='train.log')
        logger.info('Start trainning......')
        EPOCHS = 50
        # 存储训练过程
        history = {'Train Loss': [], 'Train Accuracy': []}
        for epoch in range(1, EPOCHS + 1):
            # 添加一个进度条，增加可视化效果
            processBar = tqdm(self.train_dataloader, unit='step')
            self.model.train(True)
            loss = 0
            accuracy = 0
            for step, (trainImgs, labels) in enumerate(processBar):

                trainImgs = trainImgs.to(self.device)
                labels = labels.to(self.device)

                # m, s = trainImgs.mean(), trainImgs.std()
                # trainImgs = (trainImgs - m) / s

                self.model.zero_grad()
                outputs = self.model(trainImgs)
                loss = self.loss_f(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = torch.sum(predictions == labels)/labels.shape[0]
                loss.backward()

                self.optimizer.step()
                history['Train Loss'].append(loss.item())
                history['Train Accuracy'].append(accuracy.item())
                processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                        (epoch, EPOCHS, loss.item(), accuracy.item()))
                
                if step % 1000 == 0:
                    history['Train Loss'].append(loss.item())
                    history['Train Accuracy'].append(accuracy.item())

            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , EPOCHS, loss, accuracy))
            history['Train Loss'].append(loss.item())
            history['Train Accuracy'].append(accuracy.item())
            processBar.close()
        
        logger.info('Finish training!')

        # self.model.load_state_dict(best["best_model"]) # 训练结束后，将表现最佳的模型加载到model属性，后续预测可以直接使用
