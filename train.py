'''
# -*- coding: UTF-8 -*-
    @Description:        BraTS21 + UNet3D 任务全流程
    @Author:             Junyin Xiong
    @Date:               2024/04/21
    @LastEditTime:       2024/04/21 18:55:19
    @LastEditors:        Junyin Xiong
'''

from sklearn import metrics
import torch
import time
import os
import json
# 导入本地模块
from UNet import UNet3D
from inference import inference
from transforms import train_transforms, val_transforms, ConvertToMultiChannelBasedOnBratsClassesd
from readDatasets.preprocessing import tra_trans_location
from readDatasets.preprocessing import val_trans_location

from Datasets import get_dataset_from_monai, get_dataset_from_location
from train_and_eval import train_and_eval
from readDatasets import BraTS
# 导入 MONAI 模块
from monai.losses import DiceLoss

from monai.metrics import DiceMetric
from monai.data import DataLoader, decollate_batch
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,  # 离散化处理
)

from torch.utils.tensorboard import SummaryWriter


def main():
    """整个任务
    1. 读数据
    2. 数据预处理
    3. 模型训练
    4. 模型评估
    5. 模型保存
    6. 模型可视化
    7. 模型部署
    """
    ################################*** 读取文件夹路径 ***##############################################

    if os.path.exists('/mnt/g/DATASETS/'):
        directory = '/mnt/g/DATASETS/'
    else:
        directory = 'G:\\DATASETS\\'

    # root_dir = tempfile.mkdtemp() if directory is None else directory
    root_dir = directory

    """ 定义训超参数"""
    max_epochs = 1
    batch_size = 1

    ################################*** 训练前定义 ***###################################################

    """ 定义训练设备"""
    device = torch.device("cuda:0")
    if device == "cuda:0":
        VAL_AMP = True
    else:
        VAL_AMP = False

    """ 定义模型"""
    model = UNet3D(in_channels=4, num_classes=3).to(device)  # TODO：根据类别数修改

    """ 定义优化器 """
    # optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    """ 定义学习率调度器 """
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    """ 定义损失函数  DiceLoss """
    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)

    """ 定义评估指标 """
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    """ 定义后处理 """
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])  # 后处理

    """ 定义梯度缩放器 """
    scaler = torch.cuda.amp.GradScaler()  # 自动调整梯度缩放比例

    """ enable cuDNN benchmark """
    torch.backends.cudnn.benchmark = True
    ###################################*****************###################################################

    # 1. 读数据
    # 🔅 TODO:记得修改路径
    h5_dir_path = "/mnt/g/DATASETS/datasets"
    train_txt_path = os.path.join(h5_dir_path, "train_ids.txt")
    val_txt_path = os.path.join(h5_dir_path, "valid_ids.txt")

    
    choose = input("请选择数据集来源：\n1. 本地数据集\n2. Monai\n")
    while choose not in ['1', '2']:
        choose = input("输入错误，请重新输入：\n1. 本地数据集\n2. Monai\n")
        
    if choose == '1': # choose == '1', 从本地读取数据
        tra_transforms = tra_trans_location()
        valid_transforms = val_trans_location()
        train_loader, val_loader = get_dataset_from_location(h5_dir_path=h5_dir_path,
                                                            train_txt_path=train_txt_path,
                                                            val_txt_path=val_txt_path, 
                                                            tra_transforms=tra_transforms,
                                                            val_transforms=valid_transforms)
        print(
        f"成功从本地载入数据集，当前数据集大小：{len(train_loader) + len(val_loader)} \n",
        f"训练集数量：{len(train_loader)}, 验证集数量：{len(val_loader)}")
    else:       # choose =='2', 从本地读取数据
        tra_transforms = train_transforms()
        valid_transforms = val_transforms()
        train_loader, val_loader = get_dataset_from_monai(root_dir=root_dir)
        print(
        f"成功从本地载入数据集，当前数据集大小：{len(train_loader) + len(val_loader)} \n",
        f"训练集数量：{len(train_loader)}, 验证集数量：{len(val_loader)}")
        


    # 2. 数据预处理
    # 3-4 模型训练与评估
    infer = inference(model=model, VAL_AMP=VAL_AMP)
    metric_dict = train_and_eval(model, root_dir, infer, post_trans, optimizer, lr_scheduler, loss_function, scaler,
                                dice_metric, dice_metric_batch, train_loader, val_loader, max_epochs, device)

    # 5. 保存数据
    """ 保存打印各个指标列表"""
    print(metric_dict)

    result_path = './result'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        print("已成功在{os.path.abspath(./)}目录下创建文件夹：result")
    
    metric_json = f"{time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())}_metric_list.json"
    metric_json = os.path.join(result_path, metric_json)
    metric_txt = f"{time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())}_metric_list.txt"
    metric_txt = os.path.join(result_path, metric_txt)

    with open(metric_json, 'w') as f:
        json.dump(metric_dict, f)

    data_str = json.dumps(metric_dict, indent=2)
    data_lines = data_str.split("\n")

    with open(metric_txt, 'w') as f:
        for line in data_lines:
            f.write(line + '\n')

    print(f"metric写入成功,请前往{metric_txt}查看")

    # 6. 模型可视化
    """如果epoch > 50，则绘制loss和dice曲线"""
    # if max_epochs > 50:
    # plot_metric(metric_values, metric_values_tc, metric_values_wt, metric_values_et)


if __name__ == "__main__":
    main()
