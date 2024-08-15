from witin_nn.core import NoiseModel, WitinGlobalConfig, WitinLayerConfig, HandleNegInType
import torch

class GlobalConfigFactory():

    @staticmethod
    def show_defaut_config():
        '''
        展示全部参数及其默认配置
        '''
        config = WitinGlobalConfig()
        config.noise_model = NoiseModel.NORMAL             #NORMAL|MBS|SIMPLE
        config.use_quantization = False           
        config.noise_level = 0
        config.to_linear = False
        config.w_clip = None
        config.bias_row_N = 8
        config.handle_neg_in = HandleNegInType.FALSE       #FALSE|PN|Shift
        config.x_quant_bits = 8
        config.y_quant_bits = 8
        config.bias_d = torch.tensor(0)
        config.scale_x = 1
        config.scale_y = 1
        config.scale_weight = 1
        config.shift_num = 1.0
        config.use_auto_scale = True
        config.auto_scale_updata_step = 2
        return config
    
    @staticmethod
    def get_dafault_config():
        '''
        所有参数均使用默认值，默认值见用户手册
        '''
        config = WitinGlobalConfig()
        return config

    @staticmethod
    def get_float_train_torch_config():
        '''
        采用该配置进行训练等同于使用torch.nn算子进行浮点训练
        '''
        config = WitinGlobalConfig()
        config.use_quantization = False
        return config

    @staticmethod
    def get_qat_train_wtm2100_config():
        '''
        qat训练
        '''
        config = WitinGlobalConfig()
        config.use_quantization = True
        config.handle_neg_in = HandleNegInType.PN
        config.x_quant_bits = 8
        config.y_quant_bits = 8
        config.scale_x = 16
        config.scale_y = 16
        config.scale_weight = 16
        config.auto_scale_updata_step = 20
        config.use_auto_scale = True
        return config
    
    @staticmethod
    def get_qat_nat_train_wtm2100_config():
        '''
        qat及nat训练
        '''
        config = WitinGlobalConfig()
        config.noise_model = NoiseModel.NORMAL
        config.use_quantization = True
        config.noise_level = 4
        config.handle_neg_in = HandleNegInType.PN
        config.x_quant_bits = 8
        config.y_quant_bits = 8
        config.scale_x = 16
        config.scale_y = 16
        config.scale_weight = 16
        return config


class LayerConfigFactory():
    
    @staticmethod
    def get_default_config():
        config = WitinLayerConfig(G_algo=1)
        return config

    @staticmethod
    def get_my_config():
        config = WitinLayerConfig(G_algo=1)
        config.x_quant_bits = 9
        config.scale_x = 32
        return config

    ## TODO more facotry method