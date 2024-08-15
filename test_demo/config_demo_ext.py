"""
@author: zhangyixiang
Copyright 2024 Hangzhou Zhicun (Witmem) Technology Co., Ltd. All rights reserved.
"""
import sys, os
sys.path.append(os.getcwd())
from witin_nn import GlobalConfigFactory, LayerConfigFactory
from witin_nn import WitinConv2d


def solution1():
    '''
    配置方案一
    可将整网通用配置写入自定义的GlobalConfigFactory（全局配置），每个layer调用LayerConfigFactory.get_default_config()后（局部配置），将沿用全局配置。
    '''
    global_config = GlobalConfigFactory.get_float_train_wtm2101_config()  #自定GlobalConfig
    print('global_config\n',global_config)

    config1 = LayerConfigFactory.get_default_config()   #LayerConfig沿用GlobalConfig的配置
    config1.scale_weight = 1024                         #更改scale_weight参数，其他参数依旧沿用GlobalConfig的配置
    conv2d1 = WitinConv2d(in_channels=16, out_channels=3, kernel_size=(1,1), stride=(1,1), layer_config=config1)
    print('layer_config1\n',config1)


def solution2():
    '''
    配置方案二
    不定义GlobalConfigFactory（全局配置），自定义LayerConfigFactory。
    '''
    config1 = LayerConfigFactory.get_my_config()  #自定LayerConfig，LayerConfig未指定的参数将会使用默认参数
    conv2d1 = WitinConv2d(in_channels=16, out_channels=3, kernel_size=(1,1), stride=(1,1), layer_config=config1)
    print('layer_config1\n',config1)


if __name__ == '__main__':
    solution2()