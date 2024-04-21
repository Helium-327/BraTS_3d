# BraTS21 脑肿瘤分割任务
---
```log
- 2024-4-21
    @change:      changed the code format from .ipynb to .py 
    @Author:      Junyin Xiong
    @plan:        add a new reading mode 
- 2024/04/21 19:14:35
    @change:      add a new mode to read a datasets from location path 
    @Author:      Junyin Xiong
    @plan:        

```

## DATASETS
### BraTS21
#### 下载地址：

#### 数据元信息
![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/a8801cbd-13c1-4c74-a103-b3a4610adf75)

#### 图像尺寸

![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/0cab5c3f-9eeb-4549-99e0-19c25773b6fa)

#### 标签信息
![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/acce409b-c043-46e8-8556-8da09f2bfb17)

![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/561e828b-3b32-418b-8708-91fc5fe8ae31)

#### 数据描述
> ![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/5e7cefe9-50e7-4046-8a4e-3252a59e3d6e)


#### 四种模态
> ![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/7e872e46-aae7-4098-8d72-c5b6f41a9518)


标签描述
label 0：背景 (background)
label 1：坏死肿瘤核心 (necrotic tumor core，NCR）
label 2：瘤周围水肿区域 (peritumoral edema，ED)
label 3 ：增强肿瘤 (enhancing tumor，ET), (原数据集标签为4，需要手动调整)
> ![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/cb59229c-1ff6-42e1-a603-a1927e9b8d36)


## MODELS
### U-Net3D
> ![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/58236c45-a3f5-4374-b18d-d046b8e78850)


## TOOLS
### MONAI
> ![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/20c9a5b3-bb0f-412e-9cdf-7dd9ebcc58de)
>  https://monai.io/

