"""Temporal Shift Module (TSM) 实现

TSM模块通过沿时间维度移动部分通道来融合时序信息，实现高效的时序建模。
可以方便地插入到MobileNetV4等骨干网络的任意位置。

参考论文: "TSM: Temporal Shift Module for Efficient Video Understanding" (ICCV 2019)
"""
import torch
import torch.nn as nn


class TemporalShift(nn.Module):
    """时序移动模块
    
    通过部分通道的时序移动实现帧间信息交互，无需额外参数即可增强时序建模能力。
    
    Args:
        n_segment (int): 时序片段数量，通常为8或16
        fold_div (int): 参与移动的通道比例 = 1/fold_div，默认8表示移动1/8通道
    
    输入形状: (N*T, C, H, W)，其中N为batch size，T为n_segment
    输出形状: (N*T, C, H, W)
    """
    
    def __init__(self, n_segment=8, fold_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (N*T, C, H, W)
        
        Returns:
            时序移动后的张量，形状为 (N*T, C, H, W)
        """
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        
        # 如果帧数不足，直接返回原张量
        if n_batch == 0 or nt % self.n_segment != 0:
            return x
        
        # 重塑为 (N, T, C, H, W)
        x = x.view(n_batch, self.n_segment, c, h, w)
        
        # 计算需要移动的通道数
        fold = c // self.fold_div
        
        # 初始化输出，先复制原始值
        out = x.clone()
        
        # 左移：部分通道向前移动一帧（第一帧保持不变）
        if self.n_segment > 1:
            out[:, 1:, :fold] = x[:, :-1, :fold]  # 其他帧取前一帧的值
        
        # 右移：部分通道向后移动一帧（最后一帧保持不变）
        if self.n_segment > 1:
            out[:, :-1, fold:2*fold] = x[:, 1:, fold:2*fold]  # 其他帧取后一帧的值
        
        # 还原形状为 (N*T, C, H, W)
        out = out.view(nt, c, h, w)
        
        return out


class TSMConv2d(nn.Module):
    """带时序移动的卷积层包装器
    
    将TSM与Conv2d组合，在卷积前进行时序移动，实现高效的时序建模。
    可以无缝替换标准的nn.Conv2d层。
    
    Args:
        conv_layer (nn.Conv2d): 要包装的卷积层
        n_segment (int): 时序片段数量
        fold_div (int): 参与移动的通道比例 = 1/fold_div
    
    示例:
        >>> # 替换标准卷积层
        >>> conv = nn.Conv2d(64, 128, 3, padding=1)
        >>> tsm_conv = TSMConv2d(conv, n_segment=8)
        >>> x = torch.randn(16, 64, 32, 32)  # (2*8, 64, 32, 32)
        >>> out = tsm_conv(x)
    """
    
    def __init__(self, conv_layer, n_segment=8, fold_div=8):
        super().__init__()
        self.conv = conv_layer
        self.tsm = TemporalShift(n_segment=n_segment, fold_div=fold_div)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (N*T, C, H, W)
        
        Returns:
            经过时序移动和卷积的输出，形状为 (N*T, C_out, H_out, W_out)
        """
        x = self.tsm(x)
        x = self.conv(x)
        return x
    
    @property
    def weight(self):
        """访问卷积层的权重"""
        return self.conv.weight
    
    @property
    def bias(self):
        """访问卷积层的偏置"""
        return self.conv.bias


def make_tsm_conv2d(in_channels, out_channels, kernel_size, stride=1, 
                   padding=0, dilation=1, groups=1, bias=True, 
                   padding_mode='zeros', n_segment=8, fold_div=8):
    """便捷函数：创建带TSM的卷积层
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int or tuple): 卷积核大小
        stride (int or tuple): 步长
        padding (int or tuple): 填充
        dilation (int or tuple): 膨胀率
        groups (int): 分组卷积的组数
        bias (bool): 是否使用偏置
        padding_mode (str): 填充模式
        n_segment (int): 时序片段数量
        fold_div (int): 参与移动的通道比例
    
    Returns:
        TSMConv2d: 带时序移动的卷积层
    
    示例:
        >>> # 直接创建TSM卷积层
        >>> tsm_conv = make_tsm_conv2d(64, 128, 3, padding=1, n_segment=8)
    """
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, 
                     bias=bias, padding_mode=padding_mode)
    return TSMConv2d(conv, n_segment=n_segment, fold_div=fold_div)


def insert_tsm_to_module(module, n_segment=8, fold_div=8, layer_names=None):
    """在模块中插入TSM
    
    可以自动将指定层替换为带TSM的版本，或者手动指定要替换的层。
    
    Args:
        module (nn.Module): 要修改的模块
        n_segment (int): 时序片段数量
        fold_div (int): 参与移动的通道比例
        layer_names (list, optional): 要替换的层名称列表，如果为None则替换所有Conv2d
    
    返回:
        None: 原地修改模块
    
    示例:
        >>> model = nn.Sequential(
        ...     nn.Conv2d(3, 64, 3),
        ...     nn.Conv2d(64, 128, 3)
        ... )
        >>> insert_tsm_to_module(model, n_segment=8, layer_names=['0', '1'])
    """
    if layer_names is None:
        # 自动查找所有Conv2d层并替换
        for name, layer in module.named_children():
            if isinstance(layer, nn.Conv2d):
                tsm_conv = TSMConv2d(layer, n_segment=n_segment, fold_div=fold_div)
                setattr(module, name, tsm_conv)
    else:
        # 替换指定的层
        for name, child in module.named_modules():
            if name in layer_names and isinstance(child, nn.Conv2d):
                # 获取父模块和子模块名称
                parent = module
                parts = name.split('.')
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr_name = parts[-1]
                # 替换为TSM版本
                tsm_conv = TSMConv2d(child, n_segment=n_segment, fold_div=fold_div)
                setattr(parent, attr_name, tsm_conv)
