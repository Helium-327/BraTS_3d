'''
# -*- coding: UTF-8 -*-
    @Description:        本地数据预处理
    @Author:             Junyin Xiong
    @Date:               2024/04/21
    @LastEditTime:       2024/04/21 19:20:44
    @LastEditors:        Junyin Xiong
'''


import os
import torch

from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    MapTransform,
    Compose,
    Activations,
    AsDiscrete,
)


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d

def tra_trans_location():
    return Compose([
        # EnsureChannelFirstd(keys="image"),
        # convert the images to the correct data types
        EnsureTyped(keys=["image", "label"]),
        # convert the labels to multi-channel format
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        # reorient the images to RAS+ orientation
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # 重采样，使图片具有各项同性分辨率
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        # crop the images to have the same size
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        # 沿着空间轴，随机翻转
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        # 对非零像素值(nonzero)进行归一化处理
        # 归一化可以提高图像的质量和稳定性
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # 随机调整图像的亮度,调整强度为原来的10%，100%概率调整
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        # 随机强度偏移是通过在图像的每个像素上随机采样一个偏移量，然后将该像素的强度值加上这个偏移量来实现的。
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])
    
def val_trans_location():
    return Compose([
        # EnsureChannelFirstd(keys="image"),
        # convert the images to the correct data types
        EnsureTyped(keys=["image", "label"]),
        # convert the labels to multi-channel format
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        # reorient the images to RAS+ orientation
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # resample the images to have isotropic resolution
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        # normalize the intensity of the images
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])