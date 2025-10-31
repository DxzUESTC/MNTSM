## 🧩 一、数据与采样平衡

当前配置中：

```yaml
sampler:
  type: none
class_balance:
  auto_pos_weight: true
```

这确实能通过 `pos_weight` 缓解类别不平衡（假15万 : 真3万）。
 但仅用 BCE + pos_weight 在样本分布极度偏斜时容易：

- 模型输出偏向多数类；
- 指标（AUC、F1）波动大。

✅ **建议改进：**

```yaml
loss: focal_bce   # 自定义 Focal BCE，或使用 torchvision 的 FocalLoss
class_balance:
  gamma: 2.0       # 对易分类样本降权
```

或：

```yaml
sampler:
  type: balanced    # 启用 WeightedRandomSampler（每batch类别平衡）
class_balance:
  auto_pos_weight: false
```

> 实践上“balanced sampler + 普通 BCE”比单纯 pos_weight 稳定得多。

------

## ⚙️ 二、学习率与优化器

```yaml
lr: 0.001
weight_decay: 0.01
```

如果 backbone 有预训练权重（`pretrained: true`），
 那么 `lr=1e-3` 对 Mobilenet v4 + TSM 来说稍大，会导致迁移不稳、AUC 卡在 0.5~0.6。

✅ **优化建议：**

```yaml
lr: 0.0003           # 降低主干学习率
optimizer: adamw
weight_decay: 0.01
lr_scheduler:
  type: cosine
  warmup_epochs: 3
  min_lr: 0.00003
```

> AdamW + CosineAnnealing + Warmup 可显著平滑收敛曲线。
>  若验证AUC持续不升但loss稳定下降，很可能是LR过大。

------

## 🔁 三、TSM 与 MobileNet v4 适配细节

你设定：

```yaml
fold_div: 8
n_segment: 8
aggregate: mean
```

这些都合理，但有两个细节可以提升效果：

### ✅ （1） fold_div 调整

`fold_div=8` 意味着只有 1/8 通道做 temporal shift，
 可将其调为 4，在 FF++ 这类伪造细节丰富任务中能显著提升时序感知。
 （代价≈+2% FLOPs，仍非常轻量）

```yaml
fold_div: 4
```

### ✅ （2） Temporal aggregation 改进

`aggregate: mean` 会平滑掉短期伪影信号，可尝试：

```yaml
aggregate: attention
```

在 clip 级输出后加轻量 Temporal Attention pooling：
 $$
 \hat{F} = \sum_t \alpha_t F_t, \quad \alpha_t = \text{Softmax}(W^T F_t)
 $$
 这样能让模型聚焦于变化最显著的帧。

------

## ⚡ 四、显存与混合精度策略

2060S 8GB 下 batch=32、224×224、n_segment=8 较紧张。
 AMP 可以缓解显存压力，但要注意显存碎片化。

✅ 建议：

```yaml
amp: true
grad_accum_steps: 2     # 累积两步反向传播，相当于等效 batch 64
```

> 混合精度 + 梯度累积在 TSM 场景中稳定高效。

还可以设置：

```yaml
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache() 每个epoch结束后清理显存
```

------

## 🧠 五、日志与早停策略

你当前早停策略是：

```yaml
early_stop:
  metric: video_auc
  mode: max
  patience: 8
```

这很合理，但：

- 在类别不平衡时，`video_auc` 的短期波动较大；
- `clip_auc` 收敛更快，可以同时监控。

✅ 建议：

```yaml
early_stop:
  metric: video_auc
  mode: max
  patience: 10
  min_delta: 0.002
monitor_secondary: clip_auc
```

同时记录 `best_epoch` 与 `best_model_path`，方便 resume。

------

## 📊 六、附加：实验稳定性小技巧

| 项             | 建议值                                              | 说明                   |
| -------------- | --------------------------------------------------- | ---------------------- |
| 数据增强       | RandomHorizontalFlip + ColorJitter(0.2,0.2,0.2,0.1) | FF++ 模拟光照差异      |
| BatchNorm 冻结 | True（冻结前1/3层）                                 | 迁移学习收敛更稳       |
| Dropout        | 0.3                                                 | 避免过拟合少数伪造特征 |
| Clip 长度      | n_segment=8（间隔采样）                             | 每段时间跨度≈2s 较好   |
| 验证间隔       | 每2 epoch 验证                                      | 提高效率               |

------

## ✅ 最终推荐优化配置（关键段）

```yaml
lr: 0.0003
optimizer: adamw
lr_scheduler:
  type: cosine
  warmup_epochs: 3
  min_lr: 0.00003

fold_div: 4
aggregate: attention

loss: focal_bce
class_balance:
  gamma: 2.0

grad_accum_steps: 2
early_stop:
  metric: video_auc
  patience: 10
  min_delta: 0.002
```

------

是否希望我帮你把这些优化项整合成一份完整的 YAML 版本（直接可替换你现在的配置）？
 我可以把所有推荐项合并成一个精炼的“MNTSM_ResShift_MBV4_Opt.yaml”。