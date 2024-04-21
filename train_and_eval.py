import os 
import time
import torch

from torch.utils.tensorboard import SummaryWriter

# 导入本地模块
from UNet import UNet3D
from inference import inference
from transforms import train_transforms, val_transforms, ConvertToMultiChannelBasedOnBratsClassesd
from Datasets import get_dataset_from_monai
# 导入 MONAI 模块
from monai.losses import DiceLoss

from monai.metrics import DiceMetric
from monai.data import DataLoader, decollate_batch
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,             # 离散化处理
)


from torch.utils.tensorboard import SummaryWriter



from UNet import UNet3D
from training_loop import training_loop
from val_loop import val_loop

def train_and_eval(model, root_dir, inference, post_trans, optimizer, lr_scheduler, loss_function, scaler, dice_metric, dice_metric_batch, train_loader, val_loader, max_epochs, device):
    """整个训练和验证过程
    1. 遍历每个epoch
        1.1 训练
            记录 每个epoch的平均loss
            记录 每个epoch的平均metric
            
        1.2 验证
            记录 每个epoch的平均val_loss
            记录 每个epoch的平均metric
        
    Args:

    return:

    """


    
    """ 整个训练阶段的全局参数 """
    # total_step = 0                                      # 记录总步数
    best_metric = -1                                    # 记录最佳指标
    best_metric_epoch = -1                              # 记录最佳指标对应的epoch


    best_metrics_epochs_and_time = [[], [], []]         # 记录最好的性能 epochs 时间 
    train_epoch_loss_values = []                        # 记录每个epoch的平均loss
    val_epoch_loss_values = []                          # 记录每个epoch的平均val_loss

    metric_values = []                                  # 记录每个epoch的平均metric
    metric_values_tc = []                               # 记录每个epoch的平均metric_tc                             
    metric_values_wt = []                               # 记录每个epoch的平均metric_wt
    metric_values_et = []                               # 记录每个epoch的平均metric_et


    metric_dict = {
        "best_metric": best_metric,
        "metric_values": metric_values,
        "metric_values_tc": metric_values_tc,
        "metric_values_wt": metric_values_wt,
        "metric_values_et": metric_values_et,
        "train_epoch_loss_values": train_epoch_loss_values,
    }
    
    
    
    val_interval = 1
    total_start = time.time()
    best_metrics_epochs_and_time = [[], [], []]
    
    
    print(f"start training, total epochs: {max_epochs}")

    # ============== 实例化writer类，进行日志记录 ==========================
    writer = SummaryWriter("logs")  





    for epoch in range(max_epochs):
        epoch_start = time.time()
        
        print("-" * 10)
        # =============== （添加）检测模型定义 ===============================================
        print(f"{'='*10}正在检测模型初始化相关定义...{'='*10}")
        try:
            model_name = model.__class__.__name__
            print(f"当前使用的模型为：{model_name}")
        except AttributeError:
            print(f" 模型尚未定义")

        try:
            optimizer_name = optimizer.__class__.__name__
            print(f"当前使用的优化器为：{optimizer_name}")
        except AttributeError:
            print(f" 优化器尚未定义")

        try:
            lr_scheduler_name = lr_scheduler.__class__.__name__
            print(f"当前使用的学习率调度器为：{lr_scheduler_name}")
            print(f"当前的学习率调度器参数为：{lr_scheduler.get_last_lr()}")
        except AttributeError:
            print(f" 学习率调度器尚未定义")
        print(f"{'='*46}")
        # ===============================================================================
        
        """" 一个epoch """
        # ==========================================================================================
        print(f"epoch {epoch + 1}/{max_epochs}")
        loss = training_loop(model, scaler, optimizer, loss_function, train_loader, device)
        print(f"完成了第{epoch +1}个epoch的训练")
        # print(f"[{epoch + 1}/{max_epochs}] {step}/{len(train_loader) // train_loader.batch_size}"
        # f", step time: {(time.time() - step_start):.4f}"
        # )  
        print(f"train_loss: {loss.item():.4f}")     # 每个epoch结束打印损失
        metric_dict["train_epoch_loss_values"].append(loss.item())        # 保存每个epoch的平均loss
        
        # ==========================================================================================

        # 规定多少个epoch进行一次验证
        if (epoch + 1) % val_interval == 0:                 
            dice_metric, dice_metric_batch = val_loop(inference, post_trans, loss_function, dice_metric, dice_metric_batch, val_loader, device)
            print(f"完成了第{epoch+1}个epoch的验证")
            
            metric = dice_metric.aggregate().item()        
            metric_dict["metric_values"].append(metric)
            metric_batch = dice_metric_batch.aggregate()
            
            metric_tc = metric_batch[0].item()
            metric_dict["metric_values_tc"].append(metric_tc)

            metric_wt = metric_batch[1].item()
            metric_dict["metric_values_wt"].append(metric_wt)

            metric_et = metric_batch[2].item()
            metric_dict["metric_values_et"].append(metric_et)

            dice_metric.reset()
            dice_metric_batch.reset()
            
            if metric > best_metric:
                best_metric = metric
                metric_dict["best_metric"] = best_metric
                
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(root_dir, "best_metric_model.pth"),
                )
                print("saved new best metric model")
        
            print(
                # f"current epoch: {epoch + 1} current mean dice-loss: {val_loss:.4f}"
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
                )
        
        """ 记录参数 """
        
        # ====================== 打印每一步的提示 =========================================
        # writer.add_scalar("train_loss", loss.item(), total_step)

        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")

    return metric_dict
    # if max_epochs > 50:
    #     plot_metric(metric_values, metric_values_tc, metric_values_wt, metric_values_et)

    # total_time = time.time() - total_start
    
    
if __name__ == "__main__":
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
    model = UNet3D(in_channels=4,num_classes=3).to(device)  # TODO：根据类别数修改

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
    
    ################################*** 读取文件夹路径 ***##############################################
    if os.path.exists('/mnt/g/DATASETS/'):
        directory = '/mnt/g/DATASETS/'
    else:
        directory = 'G:\\DATASETS\\'
        
    # root_dir = tempfile.mkdtemp() if directory is None else directory
    root_dir = directory
    
    train_loader, val_loader= get_dataset_from_monai(root_dir=root_dir)
    
    infer = inference(model=model, VAL_AMP=VAL_AMP)
    
    train_and_eval(model, root_dir, inference, post_trans, optimizer, lr_scheduler, loss_function, scaler, dice_metric, dice_metric_batch, train_loader, val_loader, max_epochs, device)
    print("训练完成")