"""模型评估模块"""
from sklearn.metrics import roc_auc_score

def evaluate(model, dataloader):
    """在验证集上计算AUC或F1"""
    preds, labels = [], []
    # TODO: 推理逻辑
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.0
    return auc
