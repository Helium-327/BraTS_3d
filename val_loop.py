'''
# -*- coding: UTF-8 -*-
    @Description:        实现一个epoch的验证过程
    @Author:             Junyin Xiong
    @Date:               2024/04/21
    @LastEditTime:       2024/04/21 19:20:32
    @LastEditors:        Junyin Xiong
'''

from pyexpat import model
import torch
from tqdm import tqdm

# from monai.metrics import DiceMetric
from monai.data import decollate_batch
# from monai.losses import DiceCELoss


def val_loop(inference, post_trans, loss_function, dice_metric, dice_metric_batch, val_loader, device):
    """ 实现一个epoch的验证过程
    1. 将模型设置为验证模式
    2. 遍历数据集
        2.1 计算当前batch的loss
        2.2 累加每个epoch的总损失
        2.3 打印每一步的提示
        2.4 更新每个epoch的平均loss
        2.5 记录每个epoch的平均loss
        2.6 返回每个epoch的平均loss
        
    Args:
    """
    val_epoch_loss = 0
    pbar = tqdm(val_loader)
    with torch.no_grad():
    
        for val_data in pbar:
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                    )
                    # print(val_inputs.shape, val_labels.shape)  # torch.Size([1, 3, 240, 240, 155]) torch.Size([1, 3, 240, 240, 155])
                    val_outputs = inference(val_inputs)                   # model 的输入形状为 （4，240， 240 ，155）
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]  # 转换为（4，224，224，144）
                    val_outputs = val_outputs[0][None,...]
                    # print(val_outputs.shape, val_labels.shape)  # torch.Size([1, 3, 240, 240, 155]) torch.Size([1, 3, 240, 240, 155])
                    assert val_outputs.shape == val_labels.shape, f"val_outputs.shape:{val_outputs.shape} 和 val_labels.shape: {val_labels.shape} 不一致"
                    # val_loss = loss_function(val_outputs, val_labels)    # 记录每一步的val_loss
                    # val_epoch_loss += val_loss.item()
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)
                    
    return dice_metric, dice_metric_batch
