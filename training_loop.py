from torch.utils.tensorboard import SummaryWriter
import time
import torch
from tqdm import tqdm

def training_loop(model, scaler, optimizer, loss_function, train_loader, device):
    """ 实现一个epoch的训练过程
    1. 将模型设置为训练模式
    2. 遍历数据集
        2.1 计算当前batch的loss
        2.2 反向传播
        2.3 更新模型参数
        2.4 累加每个epoch的总损失
        2.5 打印每一步的提示
        2.6 更新学习率
        2.7 更新缩放器的内部状态
        2.8 更新每个epoch的平均loss
        2.9 记录每个epoch的平均loss
    3. 计算每个epoch的总损失，用以计算平均损失
    4. 更新学习率
    5. 打印每个epoch的平均loss
    6. 记录每个epoch的平均loss
    7. 返回每个epoch的平均loss
    
    Args:
        model: 训练使用的模型
        optimizer: 训练使用的优化器
        scaler: 训练使用的缩放器
        loss_function: 训练使用的损失函数
        lr_scheduler: 训练使用的学习率调度器
        train_loader: 训练数据集
        val_loader: 验证数据集
        epoch: 当前epoch
        device: 训练使用的设备
        
    Returns:
        epoch_loss: 每个epoch的总损失
        
    
    """
    # 每个epoch的参数初始化
    epoch_loss = 0
    # step = 0
    # writer = SummaryWriter("logs")
    
    # *********************** 训练阶段 ********************************************** #
    model.train()
    
    # =============================== 训练阶段参数 =================================
    """
        以下参数每个epoch都重新初始化
    """
    
    epoch_loss = 0
    step = 0
    
    pbar = tqdm(train_loader)
    
    for batch_data in pbar:
        # time.sleep(0.01)
        step_start = time.time()
        step += 1
        # total_step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        # print(inputs.shape, labels.shape)
        # ===================== 计算train_loss ===========================================
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)          # 计算每一步的训练损失
        scaler.scale(loss).backward()                      # 将损失函数缩放到适当的范围，便在计算梯度时避免数值问题
        scaler.step(optimizer)                             # 使用优化器更新模型参数
        scaler.update()                                    # 更新缩放器的内部状态
        epoch_loss += loss.item()                          # 累加每一步的训练损失，统计一个epoch的总损失，用以计算平均损失
        
        # 打印每一步的结果

        
    return loss