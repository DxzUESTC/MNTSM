"""为数据集生成训练/验证/测试集划分（按视频层面）

用法:
    python src/utils/generate_split.py --dataset Celeb-DF-v2 --seed 42
    python src/utils/generate_split.py --dataset FFPP --seed 42
"""
import pickle
import os
import sys
import argparse
from typing import Dict, Set, Tuple

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.utils.dataset_split import (
    split_clips_by_video,
    count_unique_videos,
    extract_celebdf_video_id,
    extract_celebdf_identities,
)


def _normalize_celebdf_path(path: str) -> str:
    """统一 Celeb-DF-v2 路径表示，便于匹配."""
    if not path:
        return ''
    normalized = path.strip().replace('\\', '/')
    if not normalized.lower().startswith('celeb-df-v2/'):
        normalized = f"Celeb-DF-v2/{normalized.lstrip('/')}"
    return normalized.lower()


def _load_official_test_list(file_path: str) -> Tuple[Set[str], Dict[str, int]]:
    """
    读取官方提供的测试视频列表。

    文件格式通常为：
        1 YouTube-real/00001.mp4
        0 Celeb-synthesis/id1_id0_0007.mp4

    Returns:
        test_videos: 归一化后的视频相对路径集合（含 'Celeb-DF-v2/' 前缀，统一小写）
        label_map: 该集合中每个视频的官方标签（0=伪造，1=真实，如文件中提供）
    """
    test_videos: Set[str] = set()
    label_map: Dict[str, int] = {}

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'官方测试列表不存在: {file_path}')

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith('#'):
                continue

            parts = raw.split()
            if parts[0] in {'0', '1'} and len(parts) >= 2:
                label = int(parts[0])
                rel_path = ' '.join(parts[1:])
            else:
                label = None
                rel_path = raw

            norm_path = _normalize_celebdf_path(rel_path)
            test_videos.add(norm_path)
            if label is not None:
                label_map[norm_path] = label

    return test_videos, label_map


def main():
    parser = argparse.ArgumentParser(description='为数据集生成训练/验证/测试集划分（按视频层面）')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='数据集名称（如 FFPP, Celeb-DF-v2）')
    parser.add_argument('--index_path', type=str, default='data/dataset_index.pkl',
                       help='数据集索引文件路径（默认: data/dataset_index.pkl）')
    parser.add_argument('--output_dir', type=str, default='data/splits',
                       help='划分结果保存目录（默认: data/splits）')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例（默认: 0.1，即10%%，对应8:1:1划分）')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例（默认: 0.1，即10%%，对应8:1:1划分）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认: 42）')
    parser.add_argument('--verify', action='store_true',
                       help='验证划分结果，检查是否有数据泄露')
    parser.add_argument('--test_list_path', type=str, default=None,
                       help='官方测试列表路径（如 Celeb-DF-v2/List_of_testing_videos.txt），提供时将固定测试集')
    parser.add_argument('--output_name', type=str, default=None,
                       help='输出文件名（默认: {dataset}_split_{seed}.pkl）')
    
    args = parser.parse_args()
    
    # 加载数据集索引
    if not os.path.exists(args.index_path):
        print(f"[错误] 索引文件不存在: {args.index_path}")
        sys.exit(1)
    
    print(f"[INFO] 加载数据集索引: {args.index_path}")
    with open(args.index_path, 'rb') as f:
        data = pickle.load(f)
    
    clips = data['clips'] if isinstance(data, dict) and 'clips' in data else data
    
    # 过滤指定数据集
    dataset_name_lower = args.dataset.lower()
    original_count = len(clips)
    clips = [c for c in clips if c.get('dataset_name', '').lower() == dataset_name_lower]
    
    if len(clips) == 0:
        print(f"[错误] 在索引文件中没有找到数据集 '{args.dataset}' 的 clips")
        all_datasets = set(c.get('dataset_name', 'Unknown') for c in (data['clips'] if isinstance(data, dict) else data))
        print(f"[提示] 索引文件中包含的数据集: {all_datasets}")
        sys.exit(1)
    
    print(f"[INFO] 从 {original_count} 个 clips 中筛选出 {len(clips)} 个 {args.dataset} clips")
    
    # 按视频分组统计
    video_to_clips = {}
    for clip in clips:
        video_id = clip.get('raw_rel_path', '')
        if video_id not in video_to_clips:
            video_to_clips[video_id] = []
        video_to_clips[video_id].append(clip)
    
    total_videos = len(video_to_clips)
    print(f"[INFO] 共有 {total_videos} 个唯一视频")
    
    # 统计真实和伪造视频数量
    real_videos = sum(1 for c in clips if c.get('label', 0) == 0)
    fake_videos = sum(1 for c in clips if c.get('label', 0) == 1)
    print(f"[INFO] 真实视频 clips: {real_videos}, 伪造视频 clips: {fake_videos}")

    # 如果提供了官方测试列表，优先使用官方划分
    official_test_videos = set()
    official_test_labels: Dict[str, int] = {}
    train_clips = []
    val_clips = []
    test_clips = []
    remaining_clips = clips
    applied_test_ratio = args.test_ratio

    if args.test_list_path:
        print(f"\n[INFO] 使用官方测试划分: {args.test_list_path}")
        official_test_videos, official_test_labels = _load_official_test_list(args.test_list_path)
        print(f"[INFO] 官方列表包含 {len(official_test_videos)} 个测试视频")

        test_clip_list = []
        remaining = []
        matched_videos = set()

        for clip in clips:
            raw_path = _normalize_celebdf_path(clip.get('raw_rel_path', ''))
            if raw_path in official_test_videos:
                test_clip_list.append(clip)
                matched_videos.add(raw_path)
            else:
                remaining.append(clip)

        missing_videos = official_test_videos - matched_videos
        if missing_videos:
            print(f"[WARN] {len(missing_videos)} 个官方测试视频在索引中未找到，例如: {list(missing_videos)[:5]}")
        else:
            print("[INFO] 已成功匹配所有官方测试视频到剪辑索引")

        test_clips = test_clip_list
        remaining_clips = remaining
        applied_test_ratio = 0.0  # 剩余部分仅进行训练/验证划分
        if args.test_ratio not in (0, 0.0):
            print(f"[INFO] 忽略 --test_ratio={args.test_ratio} （官方测试集已固定，剩余部分将使用 val_ratio={args.val_ratio:.3f}、test_ratio=0 进行划分）")

    # 对剩余数据执行划分
    print(f"\n[INFO] 开始划分数据集（剩余部分比例: 训练集={1-args.val_ratio-applied_test_ratio:.1%}, 验证集={args.val_ratio:.1%}, 测试集={applied_test_ratio:.1%}, seed={args.seed}）")
    train_part, val_part, extra_test_part = split_clips_by_video(
        remaining_clips,
        val_ratio=args.val_ratio,
        test_ratio=applied_test_ratio,
        seed=args.seed
    )

    train_clips.extend(train_part)
    val_clips.extend(val_part)
    if applied_test_ratio > 0:
        test_clips.extend(extra_test_part)
    elif extra_test_part:
        # 在官方测试固定但算法仍返回了少量test clips时，将其并入验证集并给出提示
        print(f"[WARN] 划分函数输出了 {len(extra_test_part)} 个额外测试clips，将其并入验证集以保持官方划分。")
        val_clips.extend(extra_test_part)
    
    # 统计划分结果
    train_videos = count_unique_videos(train_clips)
    val_videos = count_unique_videos(val_clips)
    test_videos = count_unique_videos(test_clips)

    print(f"\n[INFO] 划分结果:")
    print(f"  训练集: {len(train_clips)} clips, {train_videos} videos ({train_videos/total_videos*100:.1f}%)")
    print(f"  验证集: {len(val_clips)} clips, {val_videos} videos ({val_videos/total_videos*100:.1f}%)")
    print(f"  测试集: {len(test_clips)} clips, {test_videos} videos ({test_videos/total_videos*100:.1f}%)")
    
    # 验证数据泄露（如果启用）
    if args.verify:
        print(f"\n[INFO] 验证数据泄露...")
        train_set = set(c.get('clip_dir', '') for c in train_clips)
        val_set = set(c.get('clip_dir', '') for c in val_clips)
        test_set = set(c.get('clip_dir', '') for c in test_clips)
        
        # 1. 检查视频级别的泄露（同一视频的所有clips应在同一集合中）
        video_leak_count = 0
        for video_id, all_video_clips in video_to_clips.items():
            train_count = sum(1 for c in all_video_clips if c.get('clip_dir', '') in train_set)
            val_count = sum(1 for c in all_video_clips if c.get('clip_dir', '') in val_set)
            test_count = sum(1 for c in all_video_clips if c.get('clip_dir', '') in test_set)
            
            sets_with_clips = sum(1 for count in [train_count, val_count, test_count] if count > 0)
            
            if sets_with_clips > 1:
                video_leak_count += 1
                if video_leak_count <= 5:
                    print(f"  [视频泄露] 视频 {video_id[:80]}...")
                    print(f"              训练集: {train_count} clips, 验证集: {val_count} clips, 测试集: {test_count} clips")
        
        # 2. 检查 ID 级别的泄露（Celeb-DF-v2：同一 ID 的真实和伪造视频应在同一集合中）
        id_leak_count = 0
        celebdf_video_id_to_videos = {}
        celebdf_identity_to_videos = {}
        for video_id, all_video_clips in video_to_clips.items():
            celebdf_id = extract_celebdf_video_id(video_id)
            if celebdf_id is not None:
                if celebdf_id not in celebdf_video_id_to_videos:
                    celebdf_video_id_to_videos[celebdf_id] = []
                celebdf_video_id_to_videos[celebdf_id].append((video_id, all_video_clips))

            identities = extract_celebdf_identities(video_id)
            for identity in identities:
                celebdf_identity_to_videos.setdefault(identity, []).append((video_id, all_video_clips))
        
        if celebdf_video_id_to_videos:
            print(f"\n[INFO] 检查 Celeb-DF-v2 主身份 ID 级别的数据泄露（共 {len(celebdf_video_id_to_videos)} 个 ID）...")
            for celebdf_id, videos in celebdf_video_id_to_videos.items():
                # 检查这个 ID 的所有视频是否在同一集合中
                id_train_videos = []
                id_val_videos = []
                id_test_videos = []
                
                for video_id, video_clips in videos:
                    # 检查这个视频的clips在哪个集合中
                    train_clips_for_video = [c for c in video_clips if c.get('clip_dir', '') in train_set]
                    val_clips_for_video = [c for c in video_clips if c.get('clip_dir', '') in val_set]
                    test_clips_for_video = [c for c in video_clips if c.get('clip_dir', '') in test_set]
                    
                    if train_clips_for_video:
                        id_train_videos.append(video_id)
                    if val_clips_for_video:
                        id_val_videos.append(video_id)
                    if test_clips_for_video:
                        id_test_videos.append(video_id)
                
                # 如果这个 ID 的视频分布在多个集合中，说明有泄露
                sets_with_videos = sum(1 for video_list in [id_train_videos, id_val_videos, id_test_videos] if len(video_list) > 0)
                
                if sets_with_videos > 1:
                    id_leak_count += 1
                    if id_leak_count <= 5:
                        print(f"  [ID泄露] ID {celebdf_id}:")
                        if id_train_videos:
                            print(f"              训练集: {len(id_train_videos)} 个视频 ({', '.join([os.path.basename(v)[:30] for v in id_train_videos[:3]])}...)")
                        if id_val_videos:
                            print(f"              验证集: {len(id_val_videos)} 个视频 ({', '.join([os.path.basename(v)[:30] for v in id_val_videos[:3]])}...)")
                        if id_test_videos:
                            print(f"              测试集: {len(id_test_videos)} 个视频 ({', '.join([os.path.basename(v)[:30] for v in id_test_videos[:3]])}...)")
        
        # 总结验证结果
        print(f"\n[INFO] 验证结果总结:")
        if video_leak_count == 0:
            print("  [OK] 视频级别：没有发现数据泄露！同一视频的所有clips都在同一个集合中。")
        else:
            print(f"  [ERROR] 视频级别：发现 {video_leak_count} 个视频存在数据泄露！")
        
        identity_leak_count = 0
        if celebdf_identity_to_videos:
            print(f"\n[INFO] 检查 Celeb-DF-v2 身份连通约束（共 {len(celebdf_identity_to_videos)} 个身份）...")
            for identity, videos in celebdf_identity_to_videos.items():
                presence_sets = set()
                train_videos = []
                val_videos = []
                test_videos = []

                for video_id, video_clips in videos:
                    in_train = any(c.get('clip_dir', '') in train_set for c in video_clips)
                    in_val = any(c.get('clip_dir', '') in val_set for c in video_clips)
                    in_test = any(c.get('clip_dir', '') in test_set for c in video_clips)

                    if in_train:
                        presence_sets.add('train')
                        train_videos.append(video_id)
                    if in_val:
                        presence_sets.add('val')
                        val_videos.append(video_id)
                    if in_test:
                        presence_sets.add('test')
                        test_videos.append(video_id)

                if len(presence_sets) > 1:
                    identity_leak_count += 1
                    if identity_leak_count <= 5:
                        print(f"  [身份泄露] 身份 {identity} 出现在多个集合:")
                        if train_videos:
                            print(f"              训练集: {len(train_videos)} 个视频 ({', '.join([os.path.basename(v)[:30] for v in train_videos[:3]])}...)")
                        if val_videos:
                            print(f"              验证集: {len(val_videos)} 个视频 ({', '.join([os.path.basename(v)[:30] for v in val_videos[:3]])}...)")
                        if test_videos:
                            print(f"              测试集: {len(test_videos)} 个视频 ({', '.join([os.path.basename(v)[:30] for v in test_videos[:3]])}...)")

        if celebdf_video_id_to_videos:
            if id_leak_count == 0:
                print("  [OK] ID 级别：没有发现数据泄露！同一 ID 的所有视频（真实和伪造）都在同一个集合中。")
            else:
                print(f"  [ERROR] ID 级别：发现 {id_leak_count} 个 ID 存在数据泄露！同一 ID 的真实和伪造视频被分到了不同集合。")

        if celebdf_identity_to_videos:
            if identity_leak_count == 0:
                print("  [OK] 身份连通：已满足任一身份仅出现在单一集合的要求。")
            else:
                print(f"  [ERROR] 身份连通：发现 {identity_leak_count} 个身份同时出现在多个集合，违反划分约束。")
    
    # 保存划分结果
    if args.output_name:
        output_filename = args.output_name
    else:
        output_filename = f"{args.dataset}_split_{args.seed}.pkl"
    output_path = os.path.join(args.output_dir, output_filename)
    os.makedirs(args.output_dir, exist_ok=True)
    
    split_data = {
        'train_clips': train_clips,
        'val_clips': val_clips,
        'test_clips': test_clips,
        'split_params': {
            'dataset_name': args.dataset,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio if not args.test_list_path else 0.0,
            'seed': args.seed,
            'total_videos': total_videos,
            'train_videos': train_videos,
            'val_videos': val_videos,
            'test_videos': test_videos,
            'official_test_list': args.test_list_path,
            'official_test_videos': len(official_test_videos) if official_test_videos else None,
        }
    }
    
    print(f"\n[INFO] 保存划分结果: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(split_data, f)
    
    print(f"[完成] 数据集划分已保存到: {output_path}")
    print(f"[提示] 在训练配置文件中使用: split_cache_path: {output_path}")


if __name__ == '__main__':
    main()

