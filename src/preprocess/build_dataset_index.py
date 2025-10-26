"""构建数据集索引文件

将预处理后的 clips 信息汇总成 npy/pkl 文件，方便训练时加载
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm


def collect_clips(meta_root):
    """递归收集所有 meta 目录下的 clip_meta.json 文件
    
    Args:
        meta_root: meta 目录根路径（例如 data/meta）
    
    Returns:
        clips: 所有 clip 的列表，每个元素是一个字典，包含：
            - video_path: 原始视频路径
            - clip_dir: clip 相对路径
            - frames: 帧信息列表
            - raw_rel_path: 原始视频相对路径（用于确定标签）
    """
    clips = []
    
    print(f"[INFO] 扫描 meta 目录: {meta_root}")
    for root, dirs, files in os.walk(meta_root):
        if 'clip_meta.json' in files:
            meta_path = os.path.join(root, 'clip_meta.json')
            
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                # 跳过失败或空的视频
                if 'error' in meta or len(meta.get('clips', [])) == 0:
                    continue
                
                # 为每个 clip 添加视频级别信息
                video_info = {
                    'raw_rel_path': meta.get('raw_rel_path', ''),
                    'video': meta.get('video', ''),
                    'num_clips': meta.get('num_clips', 0)
                }
                
                # 提取所有 clips
                for clip_info in meta.get('clips', []):
                    clip_data = {
                        **video_info,
                        'clip_id': clip_info.get('clip_id', 0),
                        'clip_dir': clip_info.get('clip_dir', ''),
                        'frames': clip_info.get('frames', [])
                    }
                    clips.append(clip_data)
                    
            except Exception as e:
                print(f"[WARN] 读取失败 {meta_path}: {e}")
                continue
    
    return clips


def determine_label_from_path(rel_path):
    """根据原始视频路径确定标签
    
    Args:
        rel_path: 原始视频相对路径（例如 FFPP/deepfakes/c23/videos/xxx.mp4）
    
    Returns:
        label: 0 表示真实视频，1 表示伪造视频
    """
    # 如果路径中包含 original，则为真实视频
    if 'original' in rel_path.lower():
        return 0
    # 其他情况（deepfakes, face2face等）为伪造视频
    else:
        return 1


def organize_dataset(clips, output_root):
    """组织数据集，将 clips 分类为训练集和验证集
    
    Args:
        clips: 所有 clip 的列表
        output_root: 输出根目录
    
    Returns:
        dataset_info: 包含训练和验证数据的字典
    """
    # 统计信息
    stats = {
        'original': 0,
        'fake': 0,
        'total_clips': len(clips)
    }
    
    # 为每个 clip 分配标签
    labeled_clips = []
    for clip in tqdm(clips, desc="分配标签"):
        label = determine_label_from_path(clip['raw_rel_path'])
        clip['label'] = label
        
        if label == 0:
            stats['original'] += 1
        else:
            stats['fake'] += 1
        
        labeled_clips.append(clip)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("数据集统计:")
    print(f"  总clip数: {stats['total_clips']}")
    print(f"  真实视频clips: {stats['original']}")
    print(f"  伪造视频clips: {stats['fake']}")
    print("="*60 + "\n")
    
    return labeled_clips, stats


def save_dataset_index(clips, output_root, format='pkl'):
    """保存数据集索引文件
    
    Args:
        clips: 所有 clip 的列表
        output_root: 输出根目录
        format: 保存格式 ('pkl' 或 'npy')
    """
    os.makedirs(output_root, exist_ok=True)
    
    # 组织数据结构
    dataset_data = {
        'clips': clips,
        'num_clips': len(clips),
        'clip_format': {
            'clip_dir': 'clip的目录路径',
            'frames': '帧信息列表（包含out_name）',
            'label': '标签（0=真实，1=伪造）',
            'video': '原始视频路径'
        }
    }
    
    if format == 'pkl':
        output_path = os.path.join(output_root, 'dataset_index.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(dataset_data, f)
        print(f"[INFO] 已保存索引文件: {output_path}")
    
    elif format == 'npy':
        output_path = os.path.join(output_root, 'dataset_index.npy')
        np.save(output_path, dataset_data, allow_pickle=True)
        print(f"[INFO] 已保存索引文件: {output_path}")
    
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    # 同时保存一个可读的 JSON 统计文件
    stats_path = os.path.join(output_root, 'dataset_stats.json')
    stats = {
        'num_clips': len(clips),
        'num_fake_clips': sum(1 for c in clips if c.get('label') == 1),
        'num_real_clips': sum(1 for c in clips if c.get('label') == 0),
        'sample_clips': clips[:5] if len(clips) > 0 else []
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存统计文件: {stats_path}")
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='构建数据集索引文件')
    parser.add_argument('--meta_root', type=str, default='data/meta',
                        help='meta 目录路径')
    parser.add_argument('--output_root', type=str, default='data',
                        help='输出根目录')
    parser.add_argument('--format', type=str, default='pkl', choices=['pkl', 'npy'],
                        help='输出格式 (pkl 或 npy)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.meta_root):
        print(f"[ERROR] meta 目录不存在: {args.meta_root}")
        return
    
    # 1. 收集所有 clips
    clips = collect_clips(args.meta_root)
    
    if len(clips) == 0:
        print("[WARN] 未找到任何 clips")
        return
    
    print(f"[INFO] 找到 {len(clips)} 个 clips")
    
    # 2. 组织数据集并分配标签
    labeled_clips, stats = organize_dataset(clips, args.output_root)
    
    # 3. 保存索引文件
    save_dataset_index(labeled_clips, args.output_root, format=args.format)
    
    print("\n[SUCCESS] 数据集索引构建完成!")


if __name__ == '__main__':
    main()

