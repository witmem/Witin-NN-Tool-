import sys, os
sys.path.append(os.getcwd())
import torch
from witin_nn import WitinConv2d, WitinLinear, WitinConvTranspose2d

def get_scale(q_bit, max):
    quat_max = 2**(q_bit-1)
    max = 2 ** torch.log2(max).ceil()
    assert max != 0, "Unable to determine quant scale, because max = 0"
    scale = quat_max / max
    return scale

def main(bias_row_N, model_path, save_path):
    '''
    Args:
        bias_row_N (int):  array使用的bias行数, 需和witin_nn训练时该参数的配置一致, 默认值为8
        model_path (str):  QAT/NAT训练完的浮点模型路径
        save_path (str):   转换后的定点模型保存路径

    Note:
        用户自行定义模型
    '''
    net = your_model()                                                        #define your model
    net.load_state_dict(torch.load(model_path), strict=False)                 #load your weights
    target_array_module = [WitinConvTranspose2d, WitinConv2d, WitinLinear]
    target_other_module = []

    array_mo_list = []
    for tar_mo in target_array_module:
        for mo in net.modules():
            if isinstance(mo, tar_mo):
                array_mo_list.append(mo)

    other_mo_list = []
    for tar_mo in target_other_module:
        for mo in net.modules():
            if isinstance(mo, tar_mo):
                other_mo_list.append(mo)

    for arr_mo in array_mo_list:
        weight_io_max = arr_mo.autoscale_weight_obj.io_max
        x_io_max = arr_mo.autoscale_x_obj.io_max

        x_scale = get_scale(9, x_io_max)
        weight_scale = get_scale(8, weight_io_max)

        arr_mo.weight.data = (arr_mo.weight.data * weight_scale).round().clip(-128, 127)
        NPU_bias = (arr_mo.bias.data * weight_scale * x_scale /128).round().clip(-128*bias_row_N, 127*bias_row_N)*128
        arr_mo.bias.data = NPU_bias
        
    torch.save(net, save_path)

if __name__ == '__main__':
    model_path = 'classification/resnet18_qat_000/models/73.pth'
    save_path = 'classification/resnet18_qat_000/models/fixed_point.pth'
    bias_row_N = 8
    main(bias_row_N, model_path, save_path)