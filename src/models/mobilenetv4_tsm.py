"""MNTSM 主干网络定义"""
import torch.nn as nn
from .tsm import TemporalShift

class MNTSMModel(nn.Module):
    def __init__(self, backbone, n_segment=8, with_attention=False):
        super().__init__()
        self.tsm = TemporalShift(n_segment=n_segment)
        self.backbone = backbone
        self.with_attention = with_attention

    def forward(self, x):
        x = self.tsm(x)
        out = self.backbone(x)
        return out
