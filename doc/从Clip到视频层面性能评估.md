## Video-level 评估流程

### 1. Clip → Video 聚合（第119-125行）

```119:125:src/train/evaluator.py
    # video-level：对每个视频的 clip 概率做平均
    video_labels = []
    video_probs = []
    for vid, probs in video_to_probs.items():
        video_probs.append(sum(probs) / max(1, len(probs)))
        video_labels.append(video_to_labels[vid])
```

- 每个视频收集其所有 clip 的概率
- 取平均值：`sum(probs) / len(probs)`
- 视频标签来自该视频的第一个 clip（同一视频的 clips 标签一致）

### 2. AUC 计算（第160行）

```159:162:src/train/evaluator.py
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    except Exception:
        auc = 0.0
```

- 使用 `sklearn.metrics.roc_auc_score`
- 输入：所有视频的真实标签（`video_labels`）和平均概率（`video_probs`）
- 计算 ROC 曲线下面积

### 3. ACC 计算（第164-172行）

```164:172:src/train/evaluator.py
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    try:
        f1 = f1_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    except Exception:
        f1 = 0.0
    try:
        bacc = balanced_accuracy_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    except Exception:
        bacc = 0.0
```

注意：代码返回的是 `bacc`（平衡准确率），而不是普通准确率。
- 阈值：0.5（概率 ≥ 0.5 为 1，否则为 0）
- 预测：`y_pred = [1 if p >= 0.5 else 0 for p in video_probs]`
- 计算：使用 `sklearn.metrics.balanced_accuracy_score`（各类别准确率的平均值，对类别不平衡更稳健）

总结：
- AUC：基于所有视频的平均概率直接计算
- ACC（实际为 BACC）：先阈值化得到预测，再计算平衡准确率

如需普通准确率（accuracy），可在 `_compute_binary_metrics` 中添加 `from sklearn.metrics import accuracy_score`，并返回 `acc = accuracy_score(y_true, y_pred)`。