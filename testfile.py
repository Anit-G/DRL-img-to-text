from pyexpat import model
import cv2 as cv
from Helper import preprocessing, postprocess
import matplotlib.pyplot as plt
from custom_example import Model
# from easyocr.model.model import Model
from collections import OrderedDict
import torch
import torchvision

# load model (keep everything same as an ordinary model except output is basic alphabet and a space)
model = Model(input_channel=1,output_channel=256,hidden_size=256,num_class=27)
model_params = torch.load('Model/english_g2.pth',map_location='cpu')

mp = OrderedDict()
for key,values in model_params.items():
    key = key.replace('module.','')
    mp[key] = values
    # print(f"{key}: {values.shape}")
mp['Prediction.weight'] = torch.randn((27, 256)) * 0.01
mp['Prediction.bias'] = torch.zeros(27)
model.load_state_dict(mp)

# change the prediction layer
# Disable training on all layers except the final prediction layer
for param in model.parameters():
        param.requires_grad = False
for param in model.Prediction.parameters():
        param.requires_grad = True
print(model)

