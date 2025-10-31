"""模型评估模块"""
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score


def _infer_step(model, batch, device, n_segment=None, aggregate="mean", amp=False):
    """对一个 batch 进行前向并返回 clip-level 概率与标签、可选视频ID。"""
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            clip_tensor, labels, meta = batch
        else:
            clip_tensor, labels = batch
            meta = None
    else:
        raise ValueError("dataloader batch 必须是 (clip_tensor, label) 或 (clip_tensor, label, meta)")

    clip_tensor = clip_tensor.to(device, non_blocking=True)  # (B, C, T, H, W)
    labels = labels.to(device, non_blocking=True)

    B, C, T, H, W = clip_tensor.shape
    T_eff = T if n_segment is None else n_segment
    if T != T_eff:
        T_eff = T

    # (B, C, T, H, W) -> (B*T, C, H, W)
    inputs = clip_tensor.permute(0, 2, 1, 3, 4).contiguous().view(B * T_eff, C, H, W)

    with torch.no_grad():
        if amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
        else:
            outputs = model(inputs)  # (B*T, num_classes) 或 (B*T, 1)

    # 还原到 (B, T, ...)，做时序聚合
    if outputs.dim() == 2 and outputs.size(-1) > 1:
        # 多分类（假设二分类 num_classes=2），取类别1的概率
        logits = outputs.view(B, T_eff, -1)
        probs = F.softmax(logits, dim=-1)[..., 1]
        if aggregate == "max":
            clip_probs = probs.max(dim=1).values
        elif aggregate == "attention":
            scores = probs  # (B,T)
            alpha = torch.softmax(scores, dim=1)
            clip_probs = (alpha * probs).sum(dim=1)
        else:
            clip_probs = probs.mean(dim=1)
    else:
        # 二分类单通道 logits
        logits = outputs.view(B, T_eff)
        probs = torch.sigmoid(logits)
        if aggregate == "max":
            clip_probs = probs.max(dim=1).values
        elif aggregate == "attention":
            alpha = torch.softmax(logits, dim=1)  # (B,T)
            clip_probs = (alpha * probs).sum(dim=1)
        else:
            clip_probs = probs.mean(dim=1)

    # 数值稳定性：替换 NaN/Inf，并裁剪到 [0,1]
    clip_probs = torch.nan_to_num(clip_probs, nan=0.5, posinf=1.0, neginf=0.0)
    clip_probs = clip_probs.clamp_(0.0, 1.0)

    # 标签 shape: (B,)
    return clip_probs.detach().cpu(), labels.detach().cpu(), meta


def evaluate(model, dataloader, device, n_segment=None, aggregate="mean", amp=False):
    """在验证/测试集上计算 clip-level 与 video-level 指标。

    Returns:
        metrics (dict): {
            'clip_auc', 'clip_f1', 'clip_bacc', 'video_auc', 'video_f1', 'video_bacc'
        }
    """
    model.eval()

    all_clip_probs = []
    all_clip_labels = []
    video_to_probs = defaultdict(list)
    video_to_labels = {}

    for batch in dataloader:
        clip_probs, labels, meta = _infer_step(model, batch, device, n_segment=n_segment, aggregate=aggregate, amp=amp)
        all_clip_probs.extend(clip_probs.tolist())
        all_clip_labels.extend(labels.tolist())

        # 收集到 video 级：优先使用 meta 中的视频标识
        if meta is not None:
            # 允许 meta 为 list[dict] 或 dict of lists
            if isinstance(meta, (list, tuple)):
                for i, m in enumerate(meta):
                    vid = None
                    if isinstance(m, dict):
                        vid = m.get('video') or m.get('raw_rel_path') or m.get('clip_dir')
                    vid = vid if vid is not None else f"sample_{len(video_to_probs)}_{i}"
                    video_to_probs[vid].append(clip_probs[i].item())
                    video_to_labels[vid] = int(labels[i].item())
            elif isinstance(meta, dict):
                vids = meta.get('video') or meta.get('raw_rel_path') or meta.get('clip_dir')
                if isinstance(vids, (list, tuple)):
                    for i, vid in enumerate(vids):
                        vid = vid if vid is not None else f"sample_{len(video_to_probs)}_{i}"
                        video_to_probs[vid].append(clip_probs[i].item())
                        video_to_labels[vid] = int(labels[i].item())
        else:
            # 无 meta 时，退化为以样本索引作为视频ID（即 video-level == clip-level）
            base_idx = len(video_to_probs)
            for i in range(len(clip_probs)):
                vid = f"sample_{base_idx + i}"
                video_to_probs[vid].append(clip_probs[i].item())
                video_to_labels[vid] = int(labels[i].item())

    # clip-level 指标
    clip_metrics = _compute_binary_metrics(all_clip_labels, all_clip_probs)

    # video-level：对每个视频的 clip 概率做平均
    video_labels = []
    video_probs = []
    for vid, probs in video_to_probs.items():
        video_probs.append(sum(probs) / max(1, len(probs)))
        video_labels.append(video_to_labels[vid])
    video_metrics = _compute_binary_metrics(video_labels, video_probs)

    return {
        **{f"clip_{k}": v for k, v in clip_metrics.items()},
        **{f"video_{k}": v for k, v in video_metrics.items()},
    }


def _compute_binary_metrics(y_true, y_prob, threshold=0.5):
    """计算二分类的 AUC、F1、Balanced Accuracy。"""
    # 转为 list 并做数值清洗
    if not isinstance(y_prob, list):
        try:
            y_prob = list(y_prob)
        except Exception:
            y_prob = [float(p) for p in y_prob]
    cleaned = []
    for p in y_prob:
        try:
            fp = float(p)
        except Exception:
            fp = 0.5
        if fp != fp:  # NaN
            fp = 0.5
        if fp == float('inf'):
            fp = 1.0
        if fp == float('-inf'):
            fp = 0.0
        if fp < 0.0:
            fp = 0.0
        if fp > 1.0:
            fp = 1.0
        cleaned.append(fp)
    y_prob = cleaned
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    except Exception:
        auc = 0.0

    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    try:
        f1 = f1_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    except Exception:
        f1 = 0.0
    try:
        bacc = balanced_accuracy_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    except Exception:
        bacc = 0.0

    return {"auc": auc, "f1": f1, "bacc": bacc}
