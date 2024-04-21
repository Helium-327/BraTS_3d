# BraTS21 脑肿瘤分割任务

```Text
提交日志：
- 2024-4-20-
    - 完成代码框架的转换，（.ipynb -> .py）

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
> BraTS每个病例包含四个模态的磁共振成像（Magnetic Resonance Imaging，MRI），每个模态的维度为240×240×155（L×W×H）
![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/98d0305e-ddd7-481b-ab5d-6c41d40a8964)

![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/65c92764-89ab-4d08-8fee-f4bd7d6b7039)

![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/3ce7e8d1-1e69-44eb-bcc8-d5d1e840785d)

#### 四种模态
1. T1
![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/7d89f050-ebe2-48d6-b1f4-61d350d290ea)

2. T1ce
![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/859a2dfd-d43f-4be2-9f0e-552311be982f)

3. T2
![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/bba7c0de-e253-4584-bd3b-05b3fd40fb41)

4. FLAIR
![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/6df59e21-c4e9-459d-8d22-bfd862fafabb)

标签描述
label 0：背景 (background)
label 1：坏死肿瘤核心 (necrotic tumor core，NCR）
label 2：瘤周围水肿区域 (peritumoral edema，ED)
label 3 ：增强肿瘤 (enhancing tumor，ET), (原数据集标签为4，需要手动调整)
![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/cb59229c-1ff6-42e1-a603-a1927e9b8d36)


## MODELS
### U-Net3D
![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/58236c45-a3f5-4374-b18d-d046b8e78850)


## TOOLS
### MONAI
> ![image](https://github.com/Helium-327/BraTS_3d/assets/48973653/20c9a5b3-bb0f-412e-9cdf-7dd9ebcc58de)
> https://monai.io/

