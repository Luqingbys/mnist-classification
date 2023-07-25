from tqdm import tqdm
import torch
import torch.nn as nn
from utils.draw import drawLoss, drawAccuracy
from utils.logger import get_logger


def train(net: nn.Module, dataLoader, optimizer, lossFunc, output, device='cpu'):
    ''' 
    net: 模型
    dataLoader: 数据读取器
    optimizer: 优化器
    lossFunc: 损失函数
    output: 训练日志保存路径
    device: cpu or gpu
    '''
    logger = get_logger(path=output, filename='train.log')
    logger.info('Start trainning......')
    EPOCHS = 5
    # 存储训练过程
    history = {'Train Loss': [], 'Train Accuracy': []}
    for epoch in range(1, EPOCHS + 1):
        # 添加一个进度条，增加可视化效果
        processBar = tqdm(dataLoader, unit='step')
        net.train(True)
        loss = 0
        accuracy = 0
        for step, (trainImgs, labels) in enumerate(processBar):

            trainImgs = trainImgs.to(device)
            labels = labels.to(device)

            # m, s = trainImgs.mean(), trainImgs.std()
            # trainImgs = (trainImgs - m) / s

            net.zero_grad()
            outputs = net(trainImgs)
            loss = lossFunc(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predictions == labels)/labels.shape[0]
            loss.backward()

            optimizer.step()
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
    torch.save(net, output+'model.pkl')
    drawLoss(history['Train Loss'], output)
    drawAccuracy(history['Train Accuracy'], output)
    



    

