# 泛化性测试指南

本指南说明如何在 Celeb-DF-v2 数据集上测试在 FFPP 数据集上训练的模型。

## 快速开始

### 1. 确保数据集已划分

确保 Celeb-DF-v2 的数据集划分文件已存在：
- `data/splits/Celeb-DF-v2_split_42.pkl`

如果还没有划分，运行：
```bash
python src/utils/generate_split.py --dataset Celeb-DF-v2 --seed 42
```

### 2. 运行泛化性测试

使用以下命令运行泛化性测试：

```bash
python -m src.main --mode eval --config generalization/configs/generalization_test_config.yml
```

或者直接运行泛化性测试脚本：

```bash
python -m src.train.generalization_test --config generalization/configs/generalization_test_config.yml
```

## 输出结果

测试结果将保存在以下位置：

- **日志文件**: `generalization/experiments/logs/Generalization_FFPP_to_CelebDFv2.log`
- **结果文件**: `generalization/experiments/checkpoints/generalization_results.txt`

结果文件包含：
- Clip-level 指标：`clip_auc`, `clip_f1`, `clip_bacc`
- Video-level 指标：`video_auc`, `video_f1`, `video_bacc`

## 配置文件说明

配置文件位于 `generalization/configs/generalization_test_config.yml`，主要配置项：

- `checkpoint_path`: 训练好的模型检查点路径
- `test_split_path`: 测试数据集划分文件路径
- `data_root`: 数据根目录
- `batch_size`: 批次大小
- `log_dir`: 日志输出目录
- `ckpt_dir`: 检查点/结果输出目录

## 注意事项

1. **模型参数自动匹配**: 脚本会自动从检查点文件中读取模型配置（如 `n_segment`, `aggregate`, `input_size` 等），确保与训练时一致。

2. **数据集要求**: 确保 Celeb-DF-v2 的数据已预处理完成，并且数据集索引文件 `data/dataset_index.pkl` 中包含 Celeb-DF-v2 的数据。

3. **设备配置**: 默认使用 GPU（如果可用），可在配置文件中设置 `use_gpu: false` 使用 CPU。

4. **批次大小**: 根据显存大小调整 `batch_size`，默认值为 32。

## 自定义测试

如果需要测试其他数据集或模型，可以：

1. 创建新的配置文件（参考 `generalization/configs/generalization_test_config.yml`）
2. 修改 `checkpoint_path` 指向你的模型
3. 修改 `test_split_path` 指向目标数据集的划分文件
4. 运行测试命令

## 示例输出

```
[INFO] 加载模型检查点: D:\Share\MNTSM_Project\MNTSM\experiments\checkpoints\FFPP\224分辨率11月4日至5日训练结果\best.pth
[INFO] 成功加载模型权重
[INFO] 加载数据集划分: data/splits/Celeb-DF-v2_split_42.pkl
[INFO] 测试集包含 1234 个 clips
[INFO] 开始评估...
============================================================
泛化性测试结果:
============================================================
clip_auc: 0.8523
clip_f1: 0.7845
clip_bacc: 0.8123
video_auc: 0.8756
video_f1: 0.8012
video_bacc: 0.8234
============================================================
[INFO] 结果已保存到: generalization/experiments/checkpoints/generalization_results.txt
```

