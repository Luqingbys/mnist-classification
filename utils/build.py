'''将原始的二进制图像文件全部读取成.png格式文件'''
from tqdm import tqdm
import os
import argparse
import torchvision
from torchvision import transforms


parser = argparse.ArgumentParser(description='将原始的二进制图像文件读取并保存为.png文件')
parser.add_argument('--dataset', type=str, default='data/MNIST/raw')
parser.add_argument('--savepath', type=str, default='data/MNIST/png/')
args = parser.parse_args()


def makeMnistPng(image_dsets, save_path):
    ''' 
    将原始的二进制图像转存为.png格式，并分类文件夹存放
    image_dsets: torchvision.datasets
    save_path: str, default('data/MNIST/png/')
    '''
    toPIL = transforms.ToPILImage()
    initial_path = save_path
    for idx in tqdm(range(10)):
        print("Making image file as .png for index {}......".format(idx))
        num_img = 0
        # dir_path = './mnist_all/'
        save_path = initial_path + str(idx)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for image, label in image_dsets:
            # image: (1, 28, 28), label: (1, )
            if label == idx:
                filename = save_path +'/mnist_' + str(num_img) + '-' + str(idx) + '.png' # 'data/MNIST/png/1/mnist_666-1.png' .png前的一个字符必定是label
                # print(image.shape, label)
                if not os.path.exists(filename):
                    pic = toPIL(image)
                    pic.save(filename)
                    # image.save(filename)
                num_img += 1
    print('Success to make MNIST PNG image files. index={}'.format(idx))


if __name__ == '__main__':
    #这个函数包括了两个操作：将图片转换为张量，以及将图片进行归一化处理
    transform = transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean = [0.5],std = [0.5])])
    path = './data/'  #数据集下载后保存的目录

    trainData = torchvision.datasets.MNIST(path, train = True,transform = transform, download = True)
    makeMnistPng(image_dsets=trainData, save_path=args.savepath)