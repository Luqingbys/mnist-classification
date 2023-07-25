from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
import numpy as np
from PIL import Image


class sampleDataset(Dataset):
    def __init__(self, df, translation=False, transforms=None) -> None:
        super().__init__()
        self.df = df
        self.translation = translation
        # self.use_label = use_label
        self.transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = 'data/MNIST/png/'
        '''获取第index份图像数据'''
        file = self.df.loc[index][0]
        class_id = self.df.loc[index][1]
        image = Image.open(path + file)
        # print('image:', type(image))

        if self.translation:
            image = np.array(image)
            # circle translation
            dxs = dys = [-1, 0, 1]
            x, y = np.random.uniform(0, 3, 2)
            dx = dxs[int(x)]
            dy = dys[int(y)]
            # print(type(image))
            image = self.translation_shift(self.translation_shift(image, dx, 0), dy, 1)

        if self.transform != None:
            image = self.transform(image)  # image: Tensor(1, 28, 28)

        return image, int(class_id)
        # return torch.tensor(image, dtype=torch.float).unsqueeze(0), int(class_id) # (1, 28, 28)

    def translation_shift(self, image, d, axis):
        if d == 0: return image
        if d == 1: index = 0
        if d == -1: index = 27
        new_img = np.roll(image, d, axis)  # 数组image在第axis维度上循环移动d个元素
        if axis == 0:
            if d == 2: new_img[1, :] = np.zeros(28)
            if d >= 1: new_img[0, :] = np.zeros(28)
            if d <= -1: new_img[27, :] = np.zeros(28)
            if d == -2: new_img[26, :] = np.zeros(28)
        if axis == 1:
            if d == 2: new_img[:, 1] = np.zeros(28)
            if d >= 1: new_img[:, 0] = np.zeros(28)
            if d <= -1: new_img[:, 27] = np.zeros(28)
            if d == -2: new_img[:, 26] = np.zeros(28)
        return new_img


def load_data(df):
    '''接收文件列表，读取全部图像为784维向量'''
    path = 'data/MNIST/png'
