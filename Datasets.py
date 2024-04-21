import os
import torch
import numpy as np
import h5py
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from readDatasets.BraTS import BraTS


from monai.utils import set_determinism
from monai.apps import DecathlonDataset
# from monai.data import DataLoader, decollate_batch





def get_dataset_from_location(h5_dir_path, train_txt_path, val_txt_path, tra_transforms=None, val_transforms=None):
    """从本地读取数据
    Args:
        h5_dir_path (str): h5文件夹路径
        train_txt_path (str): 训练集txt文件路径
        val_txt_path (str): 验证集txt文件路径
        transforms (torchvision.transforms.Compose, optional): 数据预处理操作. Defaults to None.
        
    Returns:    
        train_loader (DataLoader): 训练集数据加载器
        val_loader (DataLoader): 验证集数据加载器
    
    """
    train_ds = BraTS(h5_dir_path, train_txt_path, transforms=tra_transforms)
    val_ds = BraTS(h5_dir_path, val_txt_path, transforms=val_transforms)
    
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    return train_loader, val_loader
    
    

def get_dataset_from_monai(root_dir, download=False, tra_transfroms=None, val_transforms=None):
    """使用monai框架读取数据集
    
    Args:
        root_dir (str): 数据集根目录
        download (bool, optional): 是否下载数据集. Defaults to False.
        
    Returns:
        train_loader (DataLoader): 训练集数据加载器
        val_loader (DataLoader): 验证集数据加载器
    
    """
    
    
    # 定义数据集
    train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=tra_transfroms,
    section="training",
    download=download,
    cache_rate=0.0,
    num_workers=4, 
    )

    val_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        transform=val_transforms,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=4,
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    return train_loader, val_loader



def visualize_dataset(datasets):
    modal = ['T1', 'T1ce', 'T2', 'Flair']
    label = ['Tumor Core', 'Whole Tumor', 'Enhancing Tumor']
    
    assert isinstance(datasets, list), "输入参数必须是数据集"
    
    val_data_example = datasets[2]
    
    print(f"image shape: {val_data_example['image'].shape}")
    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"{modal[i]}")
        img = val_data_example["image"]
        plt.imshow(img.detach().cpu(), cmap="gray")
    plt.show()

    # also visualize the 3 channels label corresponding to this image
    print(f"label shape: {val_data_example['label'].shape}")
    plt.figure("label", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"channel{i}:{label[i]}")
        plt.imshow(val_data_example["label"][i, :, :, 60].detach().cpu(), cmap="gray")
    plt.show()

if __name__ == '__main__':
    from transforms import train_transforms, val_transforms
    
    """ 设置随机种子 """
    set_determinism(seed=0)

    """ 载入本地数据集 """
    if os.path.exists('/mnt/g/DATASETS/'):
        data_dir = '/mnt/g/DATASETS/BraTS21/BraTS2021_Training_Data'
        train_txt = os.path.join(data_dir, 'train_list.txt')
    else:
        data_dir = 'G:\\DATASETS\\BraTS21\\BraTS2021_Training_Data'
        train_txt = os.path.join(data_dir, 'train_list.txt')

    """ 获取文件地址 """
    with open(train_txt, 'r') as f:
        paths = [os.path.join(data_dir, line.strip()) for line in f] 

    # 获取数据集的类别名
    class_name = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))


    if os.path.exists('/mnt/g/DATASETS/'):
        directory = '/mnt/g/DATASETS/'
    else:
        directory = 'G:\\DATASETS\\'
        
    root_dir = directory
    
    train_loader, val_loader = get_dataset_from_monai(root_dir)

    


    # 创建一个数据集
    train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=train_transforms,
    section="training",
    download=False,
    cache_rate=0.0,
    num_workers=4, 
    )
    
    ds = train_ds[0]
    
    