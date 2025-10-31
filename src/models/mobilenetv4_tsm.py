"""MobileNetV4 + TSM 主干网络定义

将TSM模块集成到MobileNetV4中，实现高效的时序建模能力。
支持灵活的TSM插入位置配置。
"""
import torch.nn as nn
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available, cannot load MobileNetV4 from timm")

from .tsm import TemporalShift, TSMConv2d, insert_tsm_to_module, make_tsm_conv2d, insert_tsm_residual_shift, insert_tsm_after_expansion


class MNTSMModel(nn.Module):
    """MobileNetV4 + TSM 模型
    
    将TSM模块插入到MobileNetV4的关键位置，实现时序建模能力。
    支持多种集成方式：
    1. 在指定层位置插入TSM
    2. 替换特定层为TSM版本
    3. 灵活配置时序参数
    
    Args:
        backbone (nn.Module): MobileNetV4骨干网络
        n_segment (int): 时序片段数量，默认8（每个clip的帧数）
        fold_div (int): TSM中参与移动的通道比例，默认8（移动1/8通道）
        tsm_locations (list, optional): TSM插入位置列表，支持：
            - 'early': 在早期层插入（推荐用于浅层特征融合）
            - 'middle': 在中间层插入（推荐用于中层特征融合）
            - 'late': 在后期层插入（推荐用于深层特征融合）
            - 层名称列表: 直接指定要插入TSM的层名称（如 ['conv1', 'layer2.0.conv1']）
            - None: 使用默认策略（倒残差升维后、depthwise前）
        mode (str): TSM集成模式
            - 'default': 在倒残差块的升维后、depthwise前插入（推荐，等效于“Expansion后插入TSM”）
            - 'residual': 残差移位（仅在可残差连接的块里优先替换depthwise）
            - 'replace': 替换指定层为TSM版本（结合 tsm_locations 使用）
            - 'insert': 在指定层前插入TSM模块（占位，当前未实现专用包装）
    
    输入形状: (N*T, C, H, W)，其中N为batch size，T为n_segment
    输出形状: (N*T, C_out, H_out, W_out)
    
    示例:
        >>> import timm
        >>> # 从timm加载MobileNetV4
        >>> backbone = timm.create_model('mobilenetv4', pretrained=True)
        >>> # 创建MNTSM模型，在所有瓶颈层插入TSM
        >>> model = MNTSMModel(backbone, n_segment=8)
        >>> # 或指定插入位置
        >>> model = MNTSMModel(backbone, n_segment=8, tsm_locations=['middle', 'late'])
    """
    
    def __init__(self, backbone, n_segment=8, fold_div=8, 
                 tsm_locations=None, mode='default'):
        super().__init__()
        self.backbone = backbone
        self.n_segment = n_segment
        self.fold_div = fold_div
        
        # 插入TSM模块
        if tsm_locations is not None:
            self._insert_tsm(tsm_locations, mode)
        else:
            # 默认策略：升维后、depthwise前插入（等效于在depthwise前做TSM）
            self._insert_tsm_default(mode)
    
    def _insert_tsm(self, locations, mode='default'):
        """在指定位置插入TSM
        
        Args:
            locations: TSM插入位置列表
            mode: 'replace' 或 'insert'
        """
        if mode == 'default':
            # 无需 locations，按默认策略插入
            insert_tsm_after_expansion(
                self.backbone,
                n_segment=self.n_segment,
                fold_div=self.fold_div,
            )
        elif mode == 'residual':
            insert_tsm_residual_shift(
                self.backbone,
                n_segment=self.n_segment,
                fold_div=self.fold_div,
            )
        elif mode == 'replace':
            # 替换模式：将指定层替换为TSM版本
            if isinstance(locations, list) and all(isinstance(loc, str) for loc in locations):
                # 如果是层名称列表
                layer_names = self._find_layers_by_names(locations)
                if layer_names:
                    insert_tsm_to_module(
                        self.backbone, 
                        n_segment=self.n_segment,
                        fold_div=self.fold_div,
                        layer_names=layer_names
                    )
            elif isinstance(locations, list):
                # 如果是位置标签列表（如 ['early', 'middle']）
                self._insert_tsm_by_labels(locations)
        elif mode == 'insert':
            # 插入模式：在指定层前插入TSM模块
            self._insert_tsm_before_layers(locations)
    
    def _insert_tsm_default(self, mode='default'):
        """默认TSM插入策略：
        - default: 在倒残差块的升维后、depthwise前插入
        - residual: 残差移位
        """
        if mode == 'residual':
            insert_tsm_residual_shift(
                self.backbone,
                n_segment=self.n_segment,
                fold_div=self.fold_div,
            )
        else:
            insert_tsm_after_expansion(
                self.backbone,
                n_segment=self.n_segment,
                fold_div=self.fold_div,
            )
    
    def _insert_tsm_by_labels(self, labels):
        """根据位置标签插入TSM
        
        Args:
            labels: 位置标签列表，如 ['early', 'middle', 'late']
        """
        # 获取所有卷积层的名称
        conv_names = []
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_names.append(name)
        
        if not conv_names:
            return
        
        total_conv = len(conv_names)
        selected_layers = []
        
        for label in labels:
            if label == 'early':
                # 前1/3层
                selected_layers.extend(conv_names[:total_conv//3])
            elif label == 'middle':
                # 中间1/3层
                selected_layers.extend(conv_names[total_conv//3:2*total_conv//3])
            elif label == 'late':
                # 后1/3层
                selected_layers.extend(conv_names[2*total_conv//3:])
        
        # 去重
        selected_layers = list(set(selected_layers))
        
        if selected_layers:
            insert_tsm_to_module(
                self.backbone,
                n_segment=self.n_segment,
                fold_div=self.fold_div,
                layer_names=selected_layers
            )
    
    def _insert_tsm_before_layers(self, layer_names):
        """在指定层前插入TSM模块（不替换原层）"""
        for name in layer_names:
            parent = self._get_parent_module(name)
            if parent is not None:
                parts = name.split('.')
                attr_name = parts[-1]
                original_layer = getattr(parent, attr_name)
                
                # 创建TSM + 原层的组合
                tsm = TemporalShift(n_segment=self.n_segment, fold_div=self.fold_div)
                # 这里需要创建一个包装器来组合TSM和原层
                # 为了简化，我们这里使用替换模式
                pass  # 暂时使用替换模式
    
    def _find_layers_by_names(self, layer_names):
        """根据名称模式查找层
        
        Args:
            layer_names: 层名称或模式列表
        
        Returns:
            匹配的完整层名称列表
        """
        all_names = [name for name, _ in self.backbone.named_modules()]
        matched = []
        for pattern in layer_names:
            for name in all_names:
                if pattern in name or name.endswith(pattern):
                    matched.append(name)
        return list(set(matched))
    
    def _get_parent_module(self, layer_name):
        """获取指定层的父模块
        
        Args:
            layer_name: 层的完整名称（如 'layer1.0.conv1'）
        
        Returns:
            父模块对象，如果不存在则返回None
        """
        parts = layer_name.split('.')
        if len(parts) <= 1:
            return self.backbone
        
        try:
            parent = self.backbone
            for part in parts[:-1]:
                parent = getattr(parent, part)
            return parent
        except AttributeError:
            return None
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (N*T, C, H, W)
        
        Returns:
            模型输出，形状取决于backbone结构
        """
        out = self.backbone(x)
        return out


def create_mntsm_model(model_name='mobilenetv4', pretrained=True, 
                       n_segment=8, fold_div=8, tsm_locations=None, 
                       mode='replace', **kwargs):
    """便捷函数：创建MobileNetV4 + TSM模型
    
    Args:
        model_name (str): 模型名称，默认'mobilenetv4'
        pretrained (bool): 是否加载预训练权重
        n_segment (int): 时序片段数量
        fold_div (int): TSM通道移动比例
        tsm_locations (list, optional): TSM插入位置
        mode (str): TSM集成模式
        **kwargs: 传递给timm.create_model的其他参数
    
    Returns:
        MNTSMModel: 集成了TSM的MobileNetV4模型
    
    示例:
        >>> # 创建默认MNTSM模型
        >>> model = create_mntsm_model()
        >>> # 创建指定位置的MNTSM模型
        >>> model = create_mntsm_model(tsm_locations=['early', 'middle'])
        >>> # 创建指定层的MNTSM模型
        >>> model = create_mntsm_model(tsm_locations=['blocks.0.conv', 'blocks.1.conv'])
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm is required to load MobileNetV4. Install it with: pip install timm")
    
    # 从timm加载MobileNetV4
    backbone = timm.create_model(model_name, pretrained=pretrained, **kwargs)
    
    # 创建MNTSM模型
    model = MNTSMModel(
        backbone=backbone,
        n_segment=n_segment,
        fold_div=fold_div,
        tsm_locations=tsm_locations,
        mode=mode
    )
    
    return model
