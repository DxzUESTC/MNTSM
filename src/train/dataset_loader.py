"""Dataset 加载模块"""
import torch
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    def __init__(self, clip_list, transform=None):
        self.clip_list = clip_list
        self.transform = transform

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        item = self.clip_list[idx]
        # TODO: 加载 clip 图像并做预处理
        return item
