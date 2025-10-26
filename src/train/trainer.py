"""模型训练模块"""
import torch

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        # TODO: 前向传播与反向传播逻辑
        loss = torch.tensor(0.0)
        total_loss += loss.item()
    return total_loss / len(dataloader)
