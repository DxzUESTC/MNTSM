"""Temporal Shift Module 实现"""
import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    def __init__(self, n_segment=8, fold_div=8):
        super().__init__()
        self.fold_div = fold_div
        self.n_segment = n_segment

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]  # shift right
        out[:, :, 2*fold:] = x[:, :, 2*fold:]
        return out.view(nt, c, h, w)
