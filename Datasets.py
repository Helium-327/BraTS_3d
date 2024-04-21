import os
from matplotlib import pyplot as plt

from transforms import train_transforms, val_transforms, ConvertToMultiChannelBasedOnBratsClassesd


from monai.utils import set_determinism
from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch

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


def get_dataset_from_monai(root_dir, download=False):
    """   """
    
    """ 定义数据增强 """
    train_transform = train_transforms()
    val_transform = val_transforms()
    
    
    """ 定义数据集 """
    train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    download=download,
    cache_rate=0.0,
    num_workers=4, 
    )

    val_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=4,
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    return train_loader, val_loader

def get_dataset_from_location(root_dir):
    # TODO : 从本地读取数据集
    pass


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
    train_loader, val_loader = get_dataset_from_monai(root_dir)

    
    
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    # 创建一个数据集
    # data = [torch.randn(1, 4, 224, 224, 144) for i in range(10)]
    # labels = [torch.randn(1, 4, 224, 224, 144) for i in range(10)]
    # dataset = TensorDataset(data, labels)

    # 创建一个DataLoader
    # dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    