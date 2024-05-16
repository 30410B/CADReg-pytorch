from torch.utils.tensorboard import SummaryWriter
import sys
from torch.utils.data import DataLoader
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from Models.network124_2 import CONFIGS as CONFIGS_TM
import Models.network124_2 as TransMorph
import thop

config = CONFIGS_TM['TransMorph']
model = TransMorph.TransMorph(config)
x = torch.randn((1,2, 160, 192, 160))
flops, params = thop.profile(model, inputs=(x,))
# 多输入则为
# flops, params = thop.profile(model, inputs=(x, y, z))
flops, params = thop.clever_format([flops, params], '%.3f')
print('flops:', flops)
print('params:', params)