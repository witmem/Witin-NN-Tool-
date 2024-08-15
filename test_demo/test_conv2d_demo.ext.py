import sys, os
sys.path.append(os.getcwd())
import torch
import random
import numpy as np
from witin_nn import GlobalConfigFactory, LayerConfigFactory, HandleNegInType, NoiseModel
from witin_nn import WitinConv2d

def setRandomSeed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

setRandomSeed(20230831)


def test_forward():
    input = torch.randn(90000)*1
    input = input.reshape(-1,10,30,30)
    input.requires_grad=True

    global_config = GlobalConfigFactory.get_qat_train_wtm2100_config()
    config = LayerConfigFactory.get_default_config()
    config.scale_x = 16
    config.scale_weight = 16
    config.scale_y = 16
    print(config)
    conv2d = WitinConv2d(10,20,3,stride=(1,1), bias=True, layer_config=config) 

    conv2d.train()
    for _ in range(3):
        output = conv2d(input)
    conv2d.eval()
    output = conv2d(input)

    print('conv2d output shape:', output.shape)

if __name__ == "__main__":
    print("==================test==================")
    test_forward()
