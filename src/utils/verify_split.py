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

# 定义划分函数（直接复制代码，避免导入依赖）
import random
import math

def _count_unique_videos(clips):
    """统计clips中唯一的视频数量"""
    unique_videos = set()
    for clip in clips:
        video_id = clip.get('raw_rel_path', '')
        if video_id:
            unique_videos.add(video_id)
    return len(unique_videos)


def _split_clips_three(clips, val_ratio=0.1, test_ratio=0.1, seed=42):
    """按视频进行 训练/验证/测试 分层随机划分，确保同一视频的所有clips都在同一个集合中。"""
    assert val_ratio >= 0 and test_ratio >= 0 and (val_ratio + test_ratio) < 1.0
    
    # 按视频分组：使用 raw_rel_path 作为视频的唯一标识
    video_to_clips = {}
    for clip in clips:
        video_id = clip.get('raw_rel_path', '')
        if video_id not in video_to_clips:
            video_to_clips[video_id] = []
        video_to_clips[video_id].append(clip)
    
    # 将视频按真实/伪造分类
    real_videos = []  # 每个元素是 (video_id, clips_list, label)
    fake_videos = []
    for video_id, video_clips in video_to_clips.items():
        # 使用第一个clip的标签（同一个视频的所有clips标签应该一致）
        label = video_clips[0].get('label', 0)
        if label == 0:
            real_videos.append((video_id, video_clips, label))
        else:
            fake_videos.append((video_id, video_clips, label))
    
    # 随机打乱视频列表
    rng = random.Random(seed)
    rng.shuffle(real_videos)
    rng.shuffle(fake_videos)
    
    def split_three(lst):
        """按视频划分"""
        n_total = len(lst)
        n_val = int(math.floor(n_total * val_ratio))
        n_test = int(math.floor(n_total * test_ratio))
        n_train = max(0, n_total - n_val - n_test)
        train_part = lst[:n_train]
        val_part = lst[n_train:n_train+n_val]
        test_part = lst[n_train+n_val:n_train+n_val+n_test]
        return train_part, val_part, test_part
    
    # 分别对真实和伪造视频进行划分
    real_tr, real_va, real_te = split_three(real_videos)
    fake_tr, fake_va, fake_te = split_three(fake_videos)
    
    # 将视频列表展平为clips列表
    def flatten_videos(video_list):
        clips_list = []
        for _, video_clips, _ in video_list:
            clips_list.extend(video_clips)
        return clips_list
    
    train_clips = flatten_videos(real_tr) + flatten_videos(fake_tr)
    val_clips = flatten_videos(real_va) + flatten_videos(fake_va)
    test_clips = flatten_videos(real_te) + flatten_videos(fake_te)
    
    # 最后打乱clips顺序（但保持视频级划分不变）
    rng.shuffle(train_clips)
    rng.shuffle(val_clips)
    rng.shuffle(test_clips)
    
    return train_clips, val_clips, test_clips

# 执行划分
train_clips, val_clips, test_clips = _split_clips_three(
    clips,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
)

print(f"\n划分结果:")
print(f"Train: {len(train_clips)} clips, {_count_unique_videos(train_clips)} videos")
print(f"Val: {len(val_clips)} clips, {_count_unique_videos(val_clips)} videos")
print(f"Test: {len(test_clips)} clips, {_count_unique_videos(test_clips)} videos")

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
train_videos = _count_unique_videos(train_clips)
val_videos = _count_unique_videos(val_clips)
test_videos = _count_unique_videos(test_clips)

print(f"\n视频划分比例:")
print(f"Train: {train_videos} / {total_videos} = {train_videos/total_videos*100:.1f}%")
print(f"Val: {val_videos} / {total_videos} = {val_videos/total_videos*100:.1f}%")
print(f"Test: {test_videos} / {total_videos} = {test_videos/total_videos*100:.1f}%")

