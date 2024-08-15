import sys, os
sys.path.append(os.getcwd())
import numpy as np
import torch
from witin_nn import WitinConv2d, WitinLinear, WitinConvTranspose2d
import json

def get_scale(q_bit, max):
    quat_max = 2**(q_bit-1)
    max = 2 ** torch.log2(max).ceil()
    assert max != 0, "Unable to determine quant scale, because max = 0"
    scale = quat_max / max
    return scale

def main(model_path, save_path):
    '''
    Args:
        model_path (str):  QAT/NAT训练完的浮点模型路径
        save_path (str):   转换后的定点模型保存路径

    Note:
        用户自行定义模型
    '''

    net = define_model()                                        #define your model
    net.load_state_dict(torch.load(model_path), strict=False)   #load your weights
    target_module = [WitinConvTranspose2d, WitinConv2d, WitinLinear]

    mo_list = []
    for tar_mo in target_module:
        for mo in net.modules():
            if isinstance(mo, tar_mo):
                mo_list.append(mo)

    json_str = {}
    for mo in mo_list:
        layer_index = mo.layer_config.index

        weight_io_max = mo.autoscale_weight_obj.io_max
        x_io_max = mo.autoscale_x_obj.io_max
        y_io_max = mo.autoscale_y_obj.io_max

        weight_scale = get_scale(8, weight_io_max)
        x_scale = get_scale(9, x_io_max)
        y_scale = get_scale(8, y_io_max)

        G_value = x_scale * weight_scale / y_scale

        json_str[str(layer_index)] = int(G_value.numpy())

    with open(save_path, 'w') as f:
        json.dump(json_str, f)

if __name__ == '__main__':
    model_path = 'classification/resnet18_qat_000/models/73.pth'
    save_path = 'test.json'
    main(model_path, save_path)