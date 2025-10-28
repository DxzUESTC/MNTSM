"""批量处理视频数据集脚本"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import yaml

from generate_clips import create_face_detector, build_clips
from build_dataset_index import collect_clips, determine_label_from_path, save_dataset_index


def load_config(config_path):
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def collect_videos(raw_root, extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """递归收集所有视频文件
    
    Args:
        raw_root: raw_videos 根目录
        extensions: 支持的视频文件扩展名
    Returns:
        video_paths: 视频文件路径列表
    """
    video_paths = []
    for root, dirs, files in os.walk(raw_root):
        for file in files:
            if file.lower().endswith(extensions):
                video_paths.append(os.path.join(root, file))
    return sorted(video_paths)


def process_single_video(args):
    """处理单个视频（用于多进程）
    
    Args:
        args: (video_path, output_root, raw_root, fps, clip_len, stride, use_gpu)
    Returns:
        (video_path, success, message)
    """
    video_path, output_root, raw_root, fps, clip_len, stride, use_gpu = args
    
    try:
        # 每个进程创建自己的 detector
        detector = create_face_detector(use_gpu=use_gpu)
        
        meta = build_clips(
            video_path=video_path,
            output_root=output_root,
            detector=detector,
            fps=fps,
            clip_len=clip_len,
            raw_root=raw_root,
            stride=stride
        )
        
        num_clips = meta.get('num_clips', 0)
        num_aligned = meta.get('num_aligned_frames', 0)
        
        return (video_path, True, f"成功: {num_clips} clips, {num_aligned} 对齐帧")
    
    except Exception as e:
        return (video_path, False, f"失败: {str(e)}")


def batch_process(raw_root, output_root, fps=4, clip_len=8, stride=None, 
                  use_gpu=True, num_workers=1, filter_pattern=None, extensions=None, 
                  skip_processed=True):
    """批量处理视频数据集
    
    Args:
        raw_root: raw_videos 根目录（例如 data/raw_videos）
        output_root: 输出根目录（例如 data/）
        fps: 抽帧帧率
        clip_len: 每个 clip 的帧数
        stride: 滑窗步长（None 表示 clip_len // 2）
        use_gpu: 是否使用 GPU
        num_workers: 并行进程数（默认为1，避免GPU竞争）
        filter_pattern: 可选的路径过滤模式（例如 'deepfakes' 只处理包含该字符串的路径）
        extensions: 支持的视频文件扩展名列表
        skip_processed: 是否跳过已处理的视频（实现断点续传）
    """
    print(f"[INFO] 收集视频文件: {raw_root}")
    if extensions is None:
        extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_paths = collect_videos(raw_root, extensions)
    
    # 过滤视频
    if filter_pattern:
        video_paths = [v for v in video_paths if filter_pattern in v]
        print(f"[INFO] 应用过滤器 '{filter_pattern}': {len(video_paths)} 个视频")
    else:
        print(f"[INFO] 找到 {len(video_paths)} 个视频")
    
    if len(video_paths) == 0:
        print("[WARN] 没有找到视频文件")
        return
    
    # 单进程模式：在主进程中创建detector并直接处理
    if num_workers == 1:
        print(f"[INFO] 使用单进程模式，GPU: {use_gpu}")
        
        # 在主进程中初始化detector（避免重复初始化）
        detector = create_face_detector(use_gpu=use_gpu)
        
        success_count = 0
        fail_count = 0
        skipped_count = 0
        
        with tqdm(total=len(video_paths), desc="处理视频") as pbar:
            for video_path in video_paths:
                try:
                    meta, status = build_clips(
                        video_path=video_path,
                        output_root=output_root,
                        detector=detector,
                        fps=fps,
                        clip_len=clip_len,
                        raw_root=raw_root,
                        stride=stride,
                        skip_if_processed=skip_processed
                    )
                    
                    if status == 'skipped':
                        skipped_count += 1
                        pbar.set_postfix_str(f"✓ {success_count} | ✗ {fail_count} | ⊘ {skipped_count}")
                    elif status == 'processed':
                        success_count += 1
                        pbar.set_postfix_str(f"✓ {success_count} | ✗ {fail_count} | ⊘ {skipped_count}")
                    else:  # failed
                        fail_count += 1
                        pbar.set_postfix_str(f"✓ {success_count} | ✗ {fail_count} | ⊘ {skipped_count}")
                    
                except Exception as e:
                    fail_count += 1
                    print(f"\n[ERROR] {os.path.basename(video_path)}: {str(e)}")
                    pbar.set_postfix_str(f"✓ {success_count} | ✗ {fail_count} | ⊘ {skipped_count}")
                
                pbar.update(1)
    else:
        # 多进程模式（不推荐，可能导致GPU竞争）
        print(f"[WARN] 使用多进程模式 ({num_workers} 进程)，可能导致GPU资源竞争")
        tasks = [(v, output_root, raw_root, fps, clip_len, stride, use_gpu) 
                 for v in video_paths]
        
        success_count = 0
        fail_count = 0
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_video, task): task[0] 
                       for task in tasks}
            
            with tqdm(total=len(video_paths), desc="处理视频") as pbar:
                for future in as_completed(futures):
                    video_path, success, message = future.result()
                    
                    if success:
                        success_count += 1
                        pbar.set_postfix_str(f"✓ {success_count} | ✗ {fail_count}")
                    else:
                        fail_count += 1
                        print(f"\n[ERROR] {os.path.basename(video_path)}: {message}")
                        pbar.set_postfix_str(f"✓ {success_count} | ✗ {fail_count}")
                    
                    pbar.update(1)
    
    print(f"\n[INFO] 处理完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    if 'skipped_count' in locals():
        print(f"  跳过: {skipped_count}")
    print(f"  总计: {len(video_paths)}")
    
    # 5) 构建数据集索引
    meta_root = os.path.join(output_root, 'meta')
    if os.path.exists(meta_root):
        print(f"\n[INFO] 开始构建数据集索引...")
        clips = collect_clips(meta_root)
        if len(clips) > 0:
            # 为每个 clip 分配标签
            for clip in clips:
                label = determine_label_from_path(clip['raw_rel_path'])
                clip['label'] = label
            
            # 保存索引文件
            index_path = save_dataset_index(clips, output_root, format='pkl')
            print(f"[INFO] 索引文件已保存: {index_path}")
            print(f"[INFO] 总共 {len(clips)} 个 clips 可用于训练")
        else:
            print("[WARN] 未找到任何 clips，跳过索引构建")


def main():
    parser = argparse.ArgumentParser(description='批量处理视频数据集生成 clips')
    
    parser.add_argument('--config', type=str, default='configs/dataset_config.yml',
                        help='配置文件路径 (默认: configs/dataset_config.yml)')
    parser.add_argument('--raw_root', type=str, default=None,
                        help='raw_videos 根目录 (会覆盖配置文件中的设置)')
    parser.add_argument('--output_root', type=str, default=None,
                        help='输出根目录 (会覆盖配置文件中的设置)')
    parser.add_argument('--fps', type=int, default=None,
                        help='抽帧帧率 (会覆盖配置文件中的设置)')
    parser.add_argument('--clip_len', type=int, default=None,
                        help='每个 clip 的帧数 (会覆盖配置文件中的设置)')
    parser.add_argument('--stride', type=int, default=None,
                        help='滑窗步长 (会覆盖配置文件中的设置)')
    parser.add_argument('--use_gpu', action='store_true',
                        help='使用 GPU (会覆盖配置文件中的设置)')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用 CPU')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='并行进程数 (会覆盖配置文件中的设置)')
    parser.add_argument('--filter', type=str, default=None,
                        help='路径过滤模式 (会覆盖配置文件中的设置)')
    parser.add_argument('--no-skip', action='store_true',
                        help='不跳过已处理的视频（默认会跳过以实现断点续传）')
    
    args = parser.parse_args()
    
    # 加载配置文件
    config = {}
    if os.path.exists(args.config):
        print(f"[INFO] 从配置文件加载参数: {args.config}")
        config = load_config(args.config)
    else:
        print(f"[WARN] 配置文件不存在: {args.config}，使用命令行参数或默认值")
    
    # 合并配置：配置文件 -> 命令行参数
    raw_root = args.raw_root or config.get('paths', {}).get('raw_root')
    output_root = args.output_root or config.get('paths', {}).get('output_root')
    fps = args.fps if args.fps is not None else config.get('video', {}).get('fps', 4)
    clip_len = args.clip_len if args.clip_len is not None else config.get('video', {}).get('clip_len', 8)
    stride = args.stride if args.stride is not None else config.get('video', {}).get('stride')
    num_workers = args.num_workers if args.num_workers is not None else config.get('device', {}).get('num_workers', 2)
    filter_pattern = args.filter if args.filter is not None else config.get('filter', {}).get('pattern')
    extensions = tuple(config.get('extensions', ['.mp4', '.avi', '.mov', '.mkv']))
    
    # 处理 GPU/CPU 设置
    use_gpu = args.use_gpu
    if not use_gpu:
        use_gpu = config.get('device', {}).get('use_gpu', True)
    if args.cpu:
        use_gpu = False
    
    # 处理 skip_processed 设置
    skip_processed = not args.no_skip
    
    # 验证必需参数
    if raw_root is None:
        print("[ERROR] raw_root 未设置，请在配置文件或命令行参数中指定")
        return
    
    if output_root is None:
        print("[ERROR] output_root 未设置，请在配置文件或命令行参数中指定")
        return
    
    # 验证路径
    if not os.path.exists(raw_root):
        print(f"[ERROR] raw_root 不存在: {raw_root}")
        return
    
    os.makedirs(output_root, exist_ok=True)
    
    # 打印配置信息
    print("\n" + "="*60)
    print("预处理配置:")
    print(f"  原始视频目录: {raw_root}")
    print(f"  输出目录: {output_root}")
    print(f"  FPS: {fps}")
    print(f"  Clip长度: {clip_len}")
    print(f"  步长: {stride if stride else '自动(clip_len // 2)'}")
    print(f"  使用GPU: {use_gpu}")
    print(f"  并行进程数: {num_workers}")
    if filter_pattern:
        print(f"  过滤模式: {filter_pattern}")
    print("="*60 + "\n")
    
    # 开始批量处理
    batch_process(
        raw_root=raw_root,
        output_root=output_root,
        fps=fps,
        clip_len=clip_len,
        stride=stride,
        use_gpu=use_gpu,
        num_workers=num_workers,
        filter_pattern=filter_pattern,
        extensions=extensions,
        skip_processed=skip_processed
    )


if __name__ == '__main__':
    # 在 Windows 上需要这个保护
    mp.set_start_method('spawn', force=True)
    main()
