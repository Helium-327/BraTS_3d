'''
# -*- coding: UTF-8 -*-
    @Description:        从本地加载BraTS数据集，必须先转换成.h5文件
    @Author:             Junyin Xiong
    @Date:               2024/04/21
    @LastEditTime:       2024/04/21 19:02:56
    @LastEditors:        Junyin Xiong
'''


import os
import h5py
import torch
import numpy as np


from torch.utils.data import Dataset

from .preprocessing import tra_trans_location, val_trans_location

class BraTS(Dataset):
    def __init__(self, h5_dir_path, txt_path, transforms=None):
        #⚠️ 在此之前，训练数据应该被转成h5文件
        """加载h5文件 
        1. 从txt文件逐行获取文件名
        2. 将文件名与路径拼接，获取h5文件路径列表
        
        
        Args:
            h5_dir_path (str): 存放h5数据文件夹路径
            txt_path (str): txt文件路径
            transform (callable, optional): 数据预处理. Defaults to None.
        """
        with open(txt_path, 'r') as f:
            self.paths = [os.path.join(h5_dir_path, line.strip()) for line in f.readlines()]
        self.transforms = transforms
        
    def __getitem__(self, item):  
        """类可索引定义
        Args:
            item (int): 索引
            
        Returns:
            dict: 包含image和label的dict
        """
        
        h5f = h5py.File(self.paths[item], 'r')
        
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        label[label == 4] = 3
        
        sample = {'image': image, 'label': label}
        
        if self.transforms:
            sample = self.transforms(sample)
        
        return sample   
    
    def __len__(self):
        return len(self.paths)
    
    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]





if __name__ == '__main__':
    h5_dir_path = '/mnt/g/DATASETS/datasets' 
    
    txt_path = os.path.join(h5_dir_path, "train_ids.txt")
    
    train_trans = tra_trans_location()
    dataset = BraTS(h5_dir_path, txt_path, transforms=train_trans)
    sample = dataset[2]
    image = sample['image']
    label = sample['label']  
    # print(np.unique(label))
    
    print(dataset[2]['image'].shape) # (4, 224, 224, 144)
    print(dataset[2]['label'].shape) # (3, 224, 224, 144)
    