"""关键帧筛选模块"""
import numpy as np

def select_keyframes(features, clip_len=8, num_segments=4):
    """
    根据特征变化选择关键帧索引（改进版）
    - 保证关键帧分布均匀
    - 支持短视频
    - 对差值归一化
    - 避免重复帧
    
    Args:
        features: 特征向量列表
        clip_len: 每个clip的帧数
        num_segments: 分段数量
    
    Returns:
        selected_indices: 选中的关键帧索引列表
    """
    n = len(features)
    if n == 0:
        return []
    
    if n <= clip_len:
        return list(range(n))  # 不足 clip_len 时直接返回所有帧

    # 计算相邻帧变化（L2差）
    diffs = np.array([0.0] + [np.linalg.norm(features[i] - features[i-1]) for i in range(1, n)])
    diffs = diffs / (np.max(diffs) + 1e-6)

    # 均分时间段，在每段中选最大变化帧
    seg_len = n // num_segments
    selected = []
    for s in range(num_segments):
        start, end = s * seg_len, (s + 1) * seg_len if s < num_segments - 1 else n
        seg_diffs = diffs[start:end]
        idx = np.argmax(seg_diffs)
        selected.append(start + idx)

    # 去重，保持顺序
    selected = sorted(list(set(selected)))
    
    # 若不足 clip_len，按特征变化程度补齐
    while len(selected) < clip_len and len(selected) < n:
        # 找到所有未选中的索引
        available = [i for i in range(n) if i not in selected]
        if len(available) == 0:
            break
        # 选择变化最大的帧
        available_diffs = [(i, diffs[i]) for i in available]
        available_diffs.sort(key=lambda x: x[1], reverse=True)
        selected.append(available_diffs[0][0])
        selected = sorted(list(set(selected)))
    
    # 最终限制为 clip_len 个
    return sorted(selected[:clip_len])
