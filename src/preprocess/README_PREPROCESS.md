# 数据预处理流程说明

## 功能概览

当前预处理脚本可以完成以下操作：

### 1. 视频数据集进行抽帧
**文件**: `extract_frames.py`

- 从视频中按指定帧率抽帧并保存
- 支持自定义帧率（默认4fps）
- 自动处理视频打开失败、FPS获取失败等异常情况
- 返回成功保存的帧数

### 2. 帧的人脸检测与五点对齐
**文件**: `face_align.py`

- 使用 InsightFace 进行人脸检测
- 五点关键点对齐（双眼、鼻尖、嘴角）
- 输出 112x112 标准对齐人脸
- 多张人脸时自动选择最大的人脸

### 3. 基于特征差计算关键帧并提取关键帧
**文件**: `keyframe_selection.py`

- 计算人脸局部区域特征（4x4网格，均值+方差）
- 基于特征变化的 L2 距离选择关键帧
- 确保关键帧分布均匀，避免重复
- 自动去重和补齐，保证输出 clip_len 个关键帧

### 4. 组织 clip，形成可训练的样本
**文件**: `generate_clips.py` + `preprocess_pipline.py` + `build_dataset_index.py`

**输出内容**:
1. **对齐后的帧图片**: `data/clip_frames/.../clip_XXX/frame.jpg`
2. **元数据JSON**: `data/meta/.../clip_meta.json`
3. **数据集索引文件**: `data/dataset_index.pkl` 或 `data/dataset_index.npy`

**索引文件格式** (`dataset_index.pkl`):
```python
{
    'clips': [
        {
            'clip_dir': 'clip_frames/FFPP/deepfakes/c23/videos/clip_000',
            'frames': [
                {'out_name': '000000.jpg', 'src_name': '000123.jpg', ...},
                ...
            ],
            'label': 1,  # 0=真实, 1=伪造
            'video': '原始视频路径',
            'raw_rel_path': 'FFPP/deepfakes/c23/videos/xxx.mp4'
        },
        ...
    ],
    'num_clips': 10000,
    'clip_format': {...}
}
```

## 完整工作流程

### 1. 配置文件设置
编辑 `configs/dataset_config.yml`:
```yaml
paths:
  raw_root: "data/raw_videos"
  output_root: "data/"
video:
  fps: 4
  clip_len: 8
  stride: null
device:
  use_gpu: true
  num_workers: 2
```

### 2. 运行预处理
```bash
python src/preprocess/preprocess_pipline.py
```

或自定义参数:
```bash
python src/preprocess/preprocess_pipline.py --config configs/dataset_config.yml --fps 6 --num_workers 4
```

### 3. 处理流程
1. **扫描视频**: 递归扫描 `data/raw_videos` 下所有视频文件
2. **多进程处理**: 使用多个worker并行处理
3. **对每个视频**:
   - 抽帧 → `data/frames/.../`
   - 人脸检测与对齐
   - 计算特征并选择关键帧
   - 生成 clips → `data/clip_frames/.../clip_XXX/`
   - 保存元数据 → `data/meta/.../clip_meta.json`
4. **构建索引**: 自动生成 `data/dataset_index.pkl`

## 输出目录结构

```
data/
├── frames/              # 原始抽帧结果
│   └── FFPP/...
├── clip_frames/         # 对齐后的 clip 帧
│   └── FFPP/
│       └── deepfakes/
│           └── c23/
│               └── videos/
│                   └── clip_000/
│                       ├── 000000.jpg
│                       ├── 000001.jpg
│                       └── ...
├── meta/                # 元数据
│   └── FFPP/...
│       └── clip_meta.json
├── dataset_index.pkl   # 数据集索引（供训练使用）
└── dataset_stats.json  # 数据集统计信息
```

## 在训练中使用

加载数据集索引:
```python
import pickle

# 加载索引文件
with open('data/dataset_index.pkl', 'rb') as f:
    dataset = pickle.load(f)

clips = dataset['clips']
print(f"总共 {len(clips)} 个 clips")

# 访问单个 clip
clip = clips[0]
print(f"标签: {clip['label']}")  # 0=真实, 1=伪造
print(f"Clip目录: {clip['clip_dir']}")
print(f"帧数: {len(clip['frames'])}")

# 加载 clip 图像
import cv2
for frame_info in clip['frames']:
    img_path = f"data/{clip['clip_dir']}/{frame_info['out_name']}"
    img = cv2.imread(img_path)
    # 处理图像...
```

## 单独生成索引

如果需要重新生成索引（例如修改了标签规则）:
```bash
python src/preprocess/build_dataset_index.py \
    --meta_root data/meta \
    --output_root data/ \
    --format pkl  # 或 npy
```

## 配置参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| fps | 抽帧帧率 | 4 |
| clip_len | 每个clip的帧数 | 8 |
| stride | 滑窗步长（长视频用） | clip_len // 2 |
| use_gpu | 是否使用GPU | true |
| num_workers | 并行进程数 | 2 |
| filter_pattern | 路径过滤（如'deepfakes'） | null |

## 注意事项

1. **GPU显存**: 使用GPU时建议 `num_workers=2-4`，避免显存不足
2. **Windows支持**: 已添加 multiprocessing.spawn 支持
3. **人脸检测失败**: 会自动跳过无人的帧
4. **视频处理失败**: 会保存错误信息到 meta，不会中断整个流程
5. **索引自动构建**: 预处理完成后自动生成，无需手动操作

