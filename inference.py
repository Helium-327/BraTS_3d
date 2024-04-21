""""
Anchor: 
    Xiong Junyin
Name: 
    inference.py
Function：
    对模型进行滑窗推理
"""

import torch
from monai.inferers import sliding_window_inference
from UNet import UNet3D

class inference():
    def __init__(self, model, VAL_AMP):
        self.model = model
        self.VAL_AMP = VAL_AMP
    def compute(self, input, model) :
        return sliding_window_inference(
            inputs=input,
            # roi_size=(240, 240, 160),
            roi_size=(224, 224, 144),       # TODO : 修改roi_size为输入图片大小
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
        
    def __call__(self, input):
        if self.VAL_AMP:
            with torch.cuda.amp.autocast():
                return self.compute(input, self.model)
        else:
            return self.compute(input, self.model)
    
    
    
# def inference(input):
#     def _compute(input):
#         return sliding_window_inference(
#             inputs=input,
#             # roi_size=(240, 240, 160),
#             roi_size=(224, 224, 144),       # TODO : 修改roi_size为输入图片大小
#             sw_batch_size=1,
#             predictor=model,
#             overlap=0.5,
#         )

#     if VAL_AMP:
#         with torch.cuda.amp.autocast():
#             return _compute(input)
#     else:
#         return _compute(input)

if __name__ == "__main__":
    VAL_AMP = True
    model = UNet3D(4, 3)
    inputs = torch.randn(1, 4, 224, 224, 144)
    infer = inference(model, VAL_AMP)
    
    outputs = infer(inputs)
    # outputs = inference(inputs)
    
    print(outputs.shape)