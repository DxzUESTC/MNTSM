"""数据集划分工具模块

提供按视频层面进行数据集划分的公共函数。
"""
import random
import math
import os
import itertools


def extract_celebdf_video_id(raw_rel_path):
    """
    从 Celeb-DF-v2 数据集的 raw_rel_path 中提取主视频 ID（用于兼容旧逻辑）。
    
    对于 Celeb-real 和 Celeb-synthesis 文件夹，提取文件名第一个下划线前的部分作为 ID。
    例如：
    - Celeb-DF-v2/Celeb-real/id0_0000.mp4 -> id0
    - Celeb-DF-v2/Celeb-synthesis/id0_id16_0000.mp4 -> id0
    
    Args:
        raw_rel_path: 视频的相对路径字符串
    
    Returns:
        video_id: 提取的视频 ID，如果不是 Celeb-real/Celeb-synthesis 则返回 None
    """
    if not raw_rel_path:
        return None

    # 检查是否是 Celeb-real 或 Celeb-synthesis
    path_lower = raw_rel_path.lower().replace('\\', '/')
    if 'celeb-real' not in path_lower and 'celeb-synthesis' not in path_lower:
        return None

    # 提取文件名
    filename = os.path.basename(raw_rel_path)
    # 提取第一个下划线前的部分作为 ID
    if '_' in filename:
        video_id = filename.split('_')[0]
        return video_id
    return None


def extract_celebdf_identities(raw_rel_path):
    """从 Celeb-DF-v2 路径中提取所有出现的身份 ID（包括被替换者）。"""
    if not raw_rel_path:
        return set()

    path_lower = raw_rel_path.lower().replace('\\', '/')
    if 'celeb-real' not in path_lower and 'celeb-synthesis' not in path_lower:
        return set()

    filename = os.path.basename(raw_rel_path)
    stem, _ = os.path.splitext(filename)
    identities = {part for part in stem.split('_') if part.startswith('id')}
    return identities


def split_clips_by_video(clips, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    按视频进行 训练/验证/测试 分层随机划分，确保同一视频的所有clips都在同一个集合中。
    
    这避免了数据泄露：
    1. 同一视频的不同clips如果在训练/验证/测试集中分布，会导致过拟合
    2. 对于 Celeb-DF-v2 数据集，同一 ID 的真实和伪造视频必须在同一集合中
    
    Args:
        clips: clip数据列表，每个clip应包含 'raw_rel_path' 和 'label' 字段
        val_ratio: 验证集比例（默认0.1，即10%）
        test_ratio: 测试集比例（默认0.1，即10%）
        seed: 随机种子（默认42）
    
    Returns:
        (train_clips, val_clips, test_clips): 三个集合的clips列表
    """
    assert val_ratio >= 0 and test_ratio >= 0 and (val_ratio + test_ratio) < 1.0
    
    # 检查是否有 Celeb-DF-v2 的 clips（需要特殊处理）
    has_celebdf = any(
        bool(extract_celebdf_identities(clip.get('raw_rel_path', ''))) 
        for clip in clips
    )
    
    if has_celebdf:
        # 对于 Celeb-DF-v2，需要按 ID 分组，而不是按 raw_rel_path
        # 首先按 raw_rel_path 分组（同一视频的所有clips）
        video_to_clips = {}
        for clip in clips:
            video_id = clip.get('raw_rel_path', '')
            if video_id not in video_to_clips:
                video_to_clips[video_id] = []
            video_to_clips[video_id].append(clip)
        
        # 然后构建身份到视频的映射
        video_to_identities = {}
        identity_to_videos = {}
        other_videos = []  # 非 Celeb-DF-v2 Celeb-real/synthesis 的视频

        for video_id, video_clips in video_to_clips.items():
            label = video_clips[0].get('label', 0)
            identities = extract_celebdf_identities(video_id)
            video_to_identities[video_id] = identities

            if identities:
                for identity in identities:
                    identity_to_videos.setdefault(identity, set()).add(video_id)
            else:
                # 其他视频（如 YouTube-real），按原来的方式处理
                other_videos.append((video_id, video_clips, label))

        # 使用身份连通分量作为划分单位，确保任意出现该身份的视频留在同一集合
        identity_components = []
        visited_identities = set()

        for identity in identity_to_videos:
            if identity in visited_identities:
                continue

            component_identities = set()
            component_videos = set()
            stack = [identity]

            while stack:
                current_identity = stack.pop()
                if current_identity in visited_identities:
                    continue

                visited_identities.add(current_identity)
                component_identities.add(current_identity)

                for video_id in identity_to_videos.get(current_identity, []):
                    if video_id not in component_videos:
                        component_videos.add(video_id)
                        # 将视频中的所有身份加入搜索，确保连通分量闭合
                        for neighbour_identity in video_to_identities.get(video_id, set()):
                            if neighbour_identity not in visited_identities:
                                stack.append(neighbour_identity)

            # 收集该连通分量所有clips以及真实/伪造数量
            all_clips = []
            real_clips_count = 0
            fake_clips_count = 0
            for video_id in component_videos:
                video_clips = video_to_clips[video_id]
                all_clips.extend(video_clips)
                label = video_clips[0].get('label', 0)
                if label == 0:
                    real_clips_count += len(video_clips)
                else:
                    fake_clips_count += len(video_clips)

            identity_components.append((component_identities, all_clips, real_clips_count, fake_clips_count, len(component_videos)))

        # 随机打乱以保证不同随机种子生成不同的候选排列
        rng = random.Random(seed)
        rng.shuffle(identity_components)
        rng.shuffle(other_videos)

        def assign_identity_components(components):
            """在满足身份连通约束的前提下，尽量逼近目标比例。"""
            if not components:
                return [], [], []

            weights = [max(comp[4], 1) for comp in components]
            real_weights = [comp[2] for comp in components]
            fake_weights = [comp[3] for comp in components]

            total_weight = sum(weights)
            train_ratio = 1 - val_ratio - test_ratio
            ratio_map = {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}

            targets = [ratio_map['train'] * total_weight,
                       ratio_map['val'] * total_weight,
                       ratio_map['test'] * total_weight]

            total_real = sum(real_weights)
            total_fake = sum(fake_weights)
            real_targets = [ratio_map['train'] * total_real,
                            ratio_map['val'] * total_real,
                            ratio_map['test'] * total_real] if total_real > 0 else [0.0, 0.0, 0.0]
            fake_targets = [ratio_map['train'] * total_fake,
                            ratio_map['val'] * total_fake,
                            ratio_map['test'] * total_fake] if total_fake > 0 else [0.0, 0.0, 0.0]

            penalty_base = total_weight ** 2
            splits = ['train', 'val', 'test']

            def score_state(weight_sums, real_sums, fake_sums):
                score = 0.0
                for idx in range(3):
                    diff = weight_sums[idx] - targets[idx]
                    score += diff * diff
                    if total_real > 0:
                        diff_real = real_sums[idx] - real_targets[idx]
                        score += diff_real * diff_real
                    if total_fake > 0:
                        diff_fake = fake_sums[idx] - fake_targets[idx]
                        score += diff_fake * diff_fake
                    if targets[idx] > 0 and weight_sums[idx] == 0:
                        score += penalty_base
                return score

            exhaustive_limit = 10
            if len(components) <= exhaustive_limit:
                best_score = float('inf')
                best_assignment = None
                for assignment in itertools.product(range(3), repeat=len(components)):
                    weight_sums = [0.0, 0.0, 0.0]
                    real_sums = [0.0, 0.0, 0.0]
                    fake_sums = [0.0, 0.0, 0.0]
                    split_lists = [[], [], []]
                    for idx, split_idx in enumerate(assignment):
                        weight_sums[split_idx] += weights[idx]
                        real_sums[split_idx] += real_weights[idx]
                        fake_sums[split_idx] += fake_weights[idx]
                        split_lists[split_idx].append(components[idx])
                    current_score = score_state(weight_sums, real_sums, fake_sums)
                    if current_score < best_score - 1e-9:
                        best_score = current_score
                        best_assignment = split_lists
                if best_assignment is None:
                    best_assignment = [[], [], []]
                return best_assignment[0], best_assignment[1], best_assignment[2]

            # 回退到贪心分配
            assigned_lists = [[], [], []]
            weight_sums = [0.0, 0.0, 0.0]
            real_sums = [0.0, 0.0, 0.0]
            fake_sums = [0.0, 0.0, 0.0]

            for idx, comp in enumerate(components):
                best_split = 0
                best_score = float('inf')
                for split_idx in range(3):
                    weight_sums[split_idx] += weights[idx]
                    real_sums[split_idx] += real_weights[idx]
                    fake_sums[split_idx] += fake_weights[idx]
                    current_score = score_state(weight_sums, real_sums, fake_sums)
                    weight_sums[split_idx] -= weights[idx]
                    real_sums[split_idx] -= real_weights[idx]
                    fake_sums[split_idx] -= fake_weights[idx]

                    if current_score < best_score - 1e-9:
                        best_score = current_score
                        best_split = split_idx
                assigned_lists[best_split].append(comp)
                weight_sums[best_split] += weights[idx]
                real_sums[best_split] += real_weights[idx]
                fake_sums[best_split] += fake_weights[idx]

            return assigned_lists[0], assigned_lists[1], assigned_lists[2]

        identity_tr, identity_va, identity_te = assign_identity_components(identity_components)

        def split_three(lst):
            """按组划分"""
            n_total = len(lst)
            n_val = int(math.floor(n_total * val_ratio))
            n_test = int(math.floor(n_total * test_ratio))
            n_train = max(0, n_total - n_val - n_test)
            train_part = lst[:n_train]
            val_part = lst[n_train:n_train+n_val]
            test_part = lst[n_train+n_val:n_train+n_val+n_test]
            return train_part, val_part, test_part

        # 其他视频也按真实/伪造分类并划分
        other_real = [v for v in other_videos if v[2] == 0]
        other_fake = [v for v in other_videos if v[2] == 1]
        rng.shuffle(other_real)
        rng.shuffle(other_fake)
        other_real_tr, other_real_va, other_real_te = split_three(other_real)
        other_fake_tr, other_fake_va, other_fake_te = split_three(other_fake)
        
        # 展平：ID 组展开为 clips，其他视频也展开
        def flatten_identity_groups(group_list):
            clips_list = []
            for _, all_clips, _, _, _ in group_list:
                clips_list.extend(all_clips)
            return clips_list
        
        def flatten_videos(video_list):
            clips_list = []
            for _, video_clips, _ in video_list:
                clips_list.extend(video_clips)
            return clips_list
        
        train_clips = (flatten_identity_groups(identity_tr) + 
                       flatten_videos(other_real_tr) + flatten_videos(other_fake_tr))
        val_clips = (flatten_identity_groups(identity_va) + 
                     flatten_videos(other_real_va) + flatten_videos(other_fake_va))
        test_clips = (flatten_identity_groups(identity_te) + 
                      flatten_videos(other_real_te) + flatten_videos(other_fake_te))
        
        # 最后打乱clips顺序（但保持视频级和ID级划分不变）
        rng.shuffle(train_clips)
        rng.shuffle(val_clips)
        rng.shuffle(test_clips)
        
        return train_clips, val_clips, test_clips
    
    else:
        # 原来的逻辑：按 raw_rel_path 分组
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


def count_unique_videos(clips):
    """统计clips中唯一的视频数量"""
    unique_videos = set()
    for clip in clips:
        video_id = clip.get('raw_rel_path', '')
        if video_id:
            unique_videos.add(video_id)
    return len(unique_videos)

