"""验证数据集划分是否避免数据泄露

检查同一视频的所有clips是否都在同一个集合中（训练/验证/测试）
"""
import pickle
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

# 加载数据集索引
index_path = 'data/dataset_index.pkl'
if not os.path.exists(index_path):
    print(f"错误：索引文件不存在: {index_path}")
    exit(1)

with open(index_path, 'rb') as f:
    data = pickle.load(f)

clips = data['clips'] if isinstance(data, dict) and 'clips' in data else data

# 只使用FFPP数据集（与训练配置一致）
clips = [c for c in clips if c.get('dataset_name', '').lower() == 'ffpp']

print(f"加载了 {len(clips)} 个 FFPP clips")

# 按视频分组
video_to_clips = {}
for clip in clips:
    video_id = clip.get('raw_rel_path', '')
    if video_id not in video_to_clips:
        video_to_clips[video_id] = []
    video_to_clips[video_id].append(clip)

print(f"共有 {len(video_to_clips)} 个唯一视频")

from src.utils.dataset_split import split_clips_by_video, count_unique_videos

# 执行划分
train_clips, val_clips, test_clips = split_clips_by_video(
    clips,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
)

print(f"\n划分结果:")
print(f"Train: {len(train_clips)} clips, {count_unique_videos(train_clips)} videos")
print(f"Val: {len(val_clips)} clips, {count_unique_videos(val_clips)} videos")
print(f"Test: {len(test_clips)} clips, {count_unique_videos(test_clips)} videos")

# 验证没有数据泄露：检查每个视频的所有clips是否都在同一个集合中
print(f"\n验证数据泄露...")

# 创建 clip 到集合的映射（使用 clip_dir 作为唯一标识，因为clips是对象）
train_set = set(c.get('clip_dir', '') for c in train_clips)
val_set = set(c.get('clip_dir', '') for c in val_clips)
test_set = set(c.get('clip_dir', '') for c in test_clips)

leak_count = 0
for video_id, all_video_clips in video_to_clips.items():
    train_count = sum(1 for c in all_video_clips if c.get('clip_dir', '') in train_set)
    val_count = sum(1 for c in all_video_clips if c.get('clip_dir', '') in val_set)
    test_count = sum(1 for c in all_video_clips if c.get('clip_dir', '') in test_set)
    
    # 如果同一个视频的clips分布在多个集合中，说明有数据泄露
    sets_with_clips = sum(1 for count in [train_count, val_count, test_count] if count > 0)
    
    if sets_with_clips > 1:
        leak_count += 1
        if leak_count <= 5:  # 只打印前5个泄露的例子
            print(f"  [泄露] 视频 {video_id[:80]}...")
            print(f"          训练集: {train_count} clips, 验证集: {val_count} clips, 测试集: {test_count} clips")

if leak_count == 0:
    print("  [OK] 没有发现数据泄露！同一视频的所有clips都在同一个集合中。")
else:
    print(f"  [ERROR] 发现 {leak_count} 个视频存在数据泄露！")

# 验证划分比例
total_videos = len(video_to_clips)
train_videos = count_unique_videos(train_clips)
val_videos = count_unique_videos(val_clips)
test_videos = count_unique_videos(test_clips)

print(f"\n视频划分比例:")
print(f"Train: {train_videos} / {total_videos} = {train_videos/total_videos*100:.1f}%")
print(f"Val: {val_videos} / {total_videos} = {val_videos/total_videos*100:.1f}%")
print(f"Test: {test_videos} / {total_videos} = {test_videos/total_videos*100:.1f}%")

