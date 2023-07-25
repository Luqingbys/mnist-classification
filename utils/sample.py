'''从MNIST数据集按一定比例随机抽样，保证各个类别样本数量相等，读取成一份.csv文件'''
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os
import argparse


parser = argparse.ArgumentParser('Sample by certain ratio from initial MNIST dataset.')
parser.add_argument('--path', type=str, default='data/MNIST/png/', help='Select a path which saves the whole png files.')
parser.add_argument('--ratio', type=float, default=0, help='Input the ratio the sample from MNIST.')
parser.add_argument('--need_split', type=bool, default=True)
args = parser.parse_args()


def get_each_nums(path):
    '''返回每一个类别的样本总数'''
    res = dict()
    # path = 'data/MNIST/png/'
    for i in range(10):
        res[str(i)] = len(os.listdir(path+str(i)))
    return res


def sample_from_mnist(ratio, path, save_rest: bool):
    '''
    按比例抽样成训练集，注意原数据集训练集一共60000份样本，每一个类别的抽取样本数量应为60000×0.1×ratio
    ratio: 比例
    '''
    
    N = 60000
    each_num = N * ratio * 0.1 if ratio != 0 else 1
    valid_each_num = int(N * 0.1 * (1-ratio) * 0.1) if ratio != 0 else int((N - 10) * 0.1 * 0.1)
    print(each_num)
    num_dict = get_each_nums(path)
    sample = []
    rest_data = []
    valid_data = [] # valid datasets
    # sample.append(np.array(['path', 'label']))
    for i in tqdm(range(10)):
        print("Sampling randomly for index {}......".format(i))
        random_list: np.ndarray = random.sample(range(num_dict[str(i)]), int(each_num)) # 注意这里要做强制类型转换，不然报错
        rest_list = np.setdiff1d(np.arange(num_dict[str(i)]), random_list)  
        valid_list: np.ndarray = random.sample(list(rest_list), valid_each_num) 
        rest_list = np.setdiff1d(rest_list, valid_list)  
        for n in random_list:
            sample.append(np.array([str(i) + '/mnist_' + str(n) + '-' + str(i) + '.png', str(i)]))
        if save_rest:
            for n in valid_list:
                valid_data.append(np.array([str(i) + '/mnist_' + str(n) + '-' + str(i) + '.png', str(i)]))
            for n in rest_list:
                rest_data.append(np.array([str(i) + '/mnist_' + str(n) + '-' + str(i) + '.png', str(i)]))
    sample = np.array(sample)
    np.random.shuffle(sample)
    np.savetxt(f'./csv/train_{str(ratio)}_classify.csv', sample, delimiter=',', fmt='%s') # 保存str类型ndarray必须加上fmt=%s
    print('Success to sample MNIST PNG image files!')

    if save_rest:
        valid_data = np.array(valid_data)
        rest_data = np.array(rest_data)
        np.random.shuffle(valid_data)
        np.random.shuffle(rest_data)
        np.savetxt(f'./csv/valid_{str(ratio)}_semi.csv', valid_data, delimiter=',', fmt='%s')
        np.savetxt(f'./csv/train_{str(ratio)}_semi.csv', rest_data, delimiter=',', fmt='%s')
        print('Success to save the rest MNIST PNG image files except sampled files!')


if __name__ == '__main__':
    # path = 'data/MNIST/png/'
    sample_from_mnist(ratio=args.ratio, path=args.path, save_rest=args.need_split)
    # print(get_each_nums(path))