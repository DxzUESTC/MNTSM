"""构建数据集索引文件

将预处理后的 clips 信息汇总成 npy/pkl 文件，方便训练时加载
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm


def collect_clips(meta_root, output_root=None, verify_clips=True, dataset_filter=None):
    """递归收集所有 meta 目录下的 clip_meta.json 文件
    
    Args:
        meta_root: meta 目录根路径（例如 data/meta）
        output_root: 输出根目录（用于验证clip目录是否存在，例如 data/）
        verify_clips: 是否验证clip目录存在（默认True）
        dataset_filter: 可选，只收集指定数据集的clips（例如 'FFPP' 或 ['FFPP', 'Celeb-DF-v2']）
                       如果为None，则收集所有数据集
    
    Returns:
        clips: 所有 clip 的列表，每个元素是一个字典，包含：
            - video_path: 原始视频路径
            - clip_dir: clip 相对路径
            - frames: 帧信息列表
            - raw_rel_path: 原始视频相对路径（用于确定标签）
            - dataset_name: 数据集名称
    """
    clips = []
    skipped_count = 0
    
    # 处理dataset_filter参数
    if dataset_filter is not None:
        if isinstance(dataset_filter, str):
            dataset_filter = [dataset_filter]
        dataset_filter = [d.lower() for d in dataset_filter]
        print(f"[INFO] 数据集过滤: {dataset_filter}")
    
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
                
                raw_rel_path = meta.get('raw_rel_path', '')
                dataset_name = extract_dataset_name(raw_rel_path)
                
                # 如果设置了数据集过滤，跳过不匹配的数据集
                if dataset_filter is not None:
                    if dataset_name.lower() not in dataset_filter:
                        continue
                
                # 为每个 clip 添加视频级别信息
                video_info = {
                    'raw_rel_path': raw_rel_path,
                    'video': meta.get('video', ''),
                    'num_clips': meta.get('num_clips', 0),
                    'dataset_name': dataset_name
                }
                
                # 提取所有 clips
                for clip_info in meta.get('clips', []):
                    clip_dir = clip_info.get('clip_dir', '')
                    
                    # 验证clip目录是否存在
                    if verify_clips and output_root:
                        clip_abs_path = os.path.join(output_root, clip_dir)
                        if not os.path.exists(clip_abs_path):
                            skipped_count += 1
                            continue
                        
                        # 验证clip目录中有帧文件
                        try:
                            if not os.path.isdir(clip_abs_path):
                                skipped_count += 1
                                continue
                            frame_files = [f for f in os.listdir(clip_abs_path) 
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            if len(frame_files) == 0:
                                skipped_count += 1
                                continue
                        except (OSError, PermissionError) as e:
                            # 如果目录无法访问（权限问题、已被删除等），跳过
                            skipped_count += 1
                            continue
                    
                    clip_data = {
                        **video_info,
                        'clip_id': clip_info.get('clip_id', 0),
                        'clip_dir': clip_dir,
                        'frames': clip_info.get('frames', [])
                    }
                    clips.append(clip_data)
                    
            except Exception as e:
                print(f"[WARN] 读取失败 {meta_path}: {e}")
                continue
    
    if skipped_count > 0:
        print(f"[INFO] 跳过了 {skipped_count} 个不存在的或空的 clip 目录")
    
    return clips


def extract_dataset_name(rel_path):
    """从路径中提取数据集名称
    
    Args:
        rel_path: 原始视频相对路径（例如 FFPP/deepfakes/c23/videos/xxx.mp4 或 Celeb-DF-v2/Celeb-real/xxx.mp4）
    
    Returns:
        dataset_name: 数据集名称（如 'FFPP', 'Celeb-DF-v2'），如果无法识别则返回 'Unknown'
    """
    if not rel_path:
        return 'Unknown'
    
    # 路径通常格式：dataset_name/subfolder/...
    parts = rel_path.replace('\\', '/').split('/')
    if len(parts) > 0:
        dataset_name = parts[0]
        # 特殊处理：Celeb-DF-v2可能包含短横线
        if 'celeb' in dataset_name.lower() and 'df' in dataset_name.lower():
            return 'Celeb-DF-v2'
        return dataset_name
    return 'Unknown'


def determine_label_from_path(rel_path):
    """根据原始视频路径确定标签
    
    Args:
        rel_path: 原始视频相对路径（例如 FFPP/deepfakes/c23/videos/xxx.mp4）
    
    Returns:
        label: 0 表示真实视频，1 表示伪造视频
    """
    rel_path_lower = rel_path.lower()
    
    # FFPP数据集：路径中包含 original，则为真实视频
    if 'original' in rel_path_lower:
        return 0
    
    # Celeb-DF-v2数据集：路径中包含 real（但不包含 synthesis），则为真实视频
    # 例如：Celeb-DF-v2/Celeb-real/xxx.mp4 或 Celeb-DF-v2/YouTube-real/xxx.mp4
    if 'real' in rel_path_lower and 'synthesis' not in rel_path_lower:
        return 0
    
    # 其他情况（deepfakes, face2face, synthesis等）为伪造视频
    return 1


def organize_dataset(clips, output_root):
    """组织数据集，将 clips 分类为训练集和验证集
    
    Args:
        clips: 所有 clip 的列表
        output_root: 输出根目录
    
    Returns:
        labeled_clips: 已分配标签的clips列表
        stats: 统计信息字典
    """
    # 按数据集分组统计
    stats_by_dataset = {}
    
    # 统计信息
    stats = {
        'original': 0,
        'fake': 0,
        'total_clips': len(clips)
    }
    
    # 为每个 clip 分配标签
    labeled_clips = []
    for clip in tqdm(clips, desc="分配标签"):
        if 'label' not in clip:
            clip['label'] = determine_label_from_path(clip['raw_rel_path'])
        
        label = clip['label']
        dataset_name = clip.get('dataset_name', 'Unknown')
        
        # 更新总统计
        if label == 0:
            stats['original'] += 1
        else:
            stats['fake'] += 1
        
        # 更新按数据集统计
        if dataset_name not in stats_by_dataset:
            stats_by_dataset[dataset_name] = {'original': 0, 'fake': 0, 'total': 0}
        
        stats_by_dataset[dataset_name]['total'] += 1
        if label == 0:
            stats_by_dataset[dataset_name]['original'] += 1
        else:
            stats_by_dataset[dataset_name]['fake'] += 1
        
        labeled_clips.append(clip)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("数据集统计:")
    print(f"  总clip数: {stats['total_clips']}")
    print(f"  真实视频clips: {stats['original']}")
    print(f"  伪造视频clips: {stats['fake']}")
    
    if len(stats_by_dataset) > 1:
        print("\n按数据集统计:")
        for dataset_name, ds_stats in stats_by_dataset.items():
            print(f"  {dataset_name}:")
            print(f"    总clips: {ds_stats['total']}")
            print(f"    真实: {ds_stats['original']}")
            print(f"    伪造: {ds_stats['fake']}")
    
    print("="*60 + "\n")
    
    return labeled_clips, stats


def save_dataset_index(clips, output_root, format='pkl', dataset_name=None, save_merged=True):
    """保存数据集索引文件
    
    Args:
        clips: 所有 clip 的列表
        output_root: 输出根目录
        format: 保存格式 ('pkl' 或 'npy')
        dataset_name: 可选，数据集名称（如 'FFPP'），如果提供则保存为单独数据集索引
                     如果为None，则保存为合并索引
        save_merged: 是否同时保存合并索引（默认True，仅当dataset_name不为None时有效）
    
    Returns:
        output_path: 保存的索引文件路径
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
            'video': '原始视频路径',
            'dataset_name': '数据集名称'
        }
    }
    
    # 如果有数据集名称，添加数据集统计信息
    if dataset_name:
        dataset_data['dataset_name'] = dataset_name
        dataset_data['num_fake_clips'] = sum(1 for c in clips if c.get('label') == 1)
        dataset_data['num_real_clips'] = sum(1 for c in clips if c.get('label') == 0)
    
    # 确定输出路径
    if dataset_name:
        # 保存单独数据集索引到 datasets/ 子目录
        datasets_dir = os.path.join(output_root, 'datasets')
        os.makedirs(datasets_dir, exist_ok=True)
        filename = f"{dataset_name}_index.{format}"
        output_path = os.path.join(datasets_dir, filename)
    else:
        # 保存合并索引到根目录
        filename = f"dataset_index.{format}"
        output_path = os.path.join(output_root, filename)
    
    # 保存索引文件
    if format == 'pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(dataset_data, f)
        print(f"[INFO] 已保存索引文件: {output_path}")
    
    elif format == 'npy':
        np.save(output_path, dataset_data, allow_pickle=True)
        print(f"[INFO] 已保存索引文件: {output_path}")
    
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    # 保存统计文件
    if dataset_name:
        stats_path = os.path.join(datasets_dir, f"{dataset_name}_stats.json")
    else:
        stats_path = os.path.join(output_root, 'dataset_stats.json')
    
    stats = {
        'num_clips': len(clips),
        'num_fake_clips': sum(1 for c in clips if c.get('label') == 1),
        'num_real_clips': sum(1 for c in clips if c.get('label') == 0),
        'sample_clips': clips[:5] if len(clips) > 0 else []
    }
    
    if dataset_name:
        stats['dataset_name'] = dataset_name
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存统计文件: {stats_path}")
    
    return output_path


def build_indices_by_dataset(meta_root, output_root, format='pkl', verify_clips=True, 
                             save_merged=True, dataset_filter=None):
    """按数据集分别构建索引，并可选择性地创建合并索引
    
    Args:
        meta_root: meta 目录根路径
        output_root: 输出根目录
        format: 保存格式 ('pkl' 或 'npy')
        verify_clips: 是否验证clip目录存在
        save_merged: 是否保存合并索引（默认True）
        dataset_filter: 可选，只处理指定数据集（例如 ['FFPP', 'Celeb-DF-v2']）
    
    Returns:
        dict: 包含每个数据集索引路径和合并索引路径的字典
    """
    # 收集所有clips
    all_clips = collect_clips(meta_root, output_root=output_root, 
                             verify_clips=verify_clips, dataset_filter=dataset_filter)
    
    if len(all_clips) == 0:
        print("[WARN] 未找到任何 clips")
        return {}
    
    # 按数据集分组
    clips_by_dataset = {}
    for clip in all_clips:
        dataset_name = clip.get('dataset_name', 'Unknown')
        if dataset_name not in clips_by_dataset:
            clips_by_dataset[dataset_name] = []
        clips_by_dataset[dataset_name].append(clip)
    
    print(f"\n[INFO] 发现 {len(clips_by_dataset)} 个数据集:")
    for dataset_name, clips in clips_by_dataset.items():
        print(f"  {dataset_name}: {len(clips)} clips")
    
    # 为每个数据集分配标签并保存单独索引
    dataset_index_paths = {}
    for dataset_name, clips in clips_by_dataset.items():
        print(f"\n[INFO] 处理数据集: {dataset_name}")
        
        # 分配标签
        for clip in clips:
            if 'label' not in clip:
                clip['label'] = determine_label_from_path(clip['raw_rel_path'])
        
        # 保存单独数据集索引
        index_path = save_dataset_index(clips, output_root, format=format, 
                                       dataset_name=dataset_name, save_merged=False)
        dataset_index_paths[dataset_name] = index_path
    
    # 保存合并索引（如果启用）
    merged_index_path = None
    if save_merged:
        print(f"\n[INFO] 构建合并索引...")
        for clip in all_clips:
            if 'label' not in clip:
                clip['label'] = determine_label_from_path(clip['raw_rel_path'])
        
        merged_index_path = save_dataset_index(all_clips, output_root, format=format,
                                              dataset_name=None, save_merged=False)
        dataset_index_paths['_merged'] = merged_index_path
    
    return dataset_index_paths


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='构建数据集索引文件')
    parser.add_argument('--meta_root', type=str, default='data/meta',
                        help='meta 目录路径')
    parser.add_argument('--output_root', type=str, default='data',
                        help='输出根目录')
    parser.add_argument('--format', type=str, default='pkl', choices=['pkl', 'npy'],
                        help='输出格式 (pkl 或 npy)')
    parser.add_argument('--no-verify', action='store_true',
                        help='不验证clip目录是否存在（默认会验证）')
    parser.add_argument('--by-dataset', action='store_true', default=True,
                        help='按数据集分别构建索引（默认启用）')
    parser.add_argument('--no-merged', action='store_true',
                        help='不创建合并索引（仅当--by-dataset时有效）')
    parser.add_argument('--dataset-filter', type=str, nargs='+', default=None,
                        help='只处理指定数据集（例如 --dataset-filter FFPP Celeb-DF-v2）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.meta_root):
        print(f"[ERROR] meta 目录不存在: {args.meta_root}")
        return
    
    if args.by_dataset:
        # 使用新的按数据集构建方式
        index_paths = build_indices_by_dataset(
            meta_root=args.meta_root,
            output_root=args.output_root,
            format=args.format,
            verify_clips=not args.no_verify,
            save_merged=not args.no_merged,
            dataset_filter=args.dataset_filter
        )
        
        if index_paths:
            print("\n[SUCCESS] 数据集索引构建完成!")
            print("\n索引文件位置:")
            for name, path in index_paths.items():
                if name == '_merged':
                    print(f"  合并索引: {path}")
                else:
                    print(f"  {name}: {path}")
        else:
            print("[WARN] 未找到任何 clips")
    else:
        # 使用旧的合并索引方式（向后兼容）
        clips = collect_clips(args.meta_root, output_root=args.output_root, 
                             verify_clips=not args.no_verify,
                             dataset_filter=args.dataset_filter)
        
        if len(clips) == 0:
            print("[WARN] 未找到任何 clips")
            return
        
        print(f"[INFO] 找到 {len(clips)} 个 clips")
        
        # 组织数据集并分配标签
        labeled_clips, stats = organize_dataset(clips, args.output_root)
        
        # 保存索引文件（合并索引）
        save_dataset_index(labeled_clips, args.output_root, format=args.format, 
                         dataset_name=None, save_merged=False)
        
        print("\n[SUCCESS] 数据集索引构建完成!")


if __name__ == '__main__':
    main()

