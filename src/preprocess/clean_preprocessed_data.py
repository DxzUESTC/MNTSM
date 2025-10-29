"""清理预处理生成的文件和目录

该脚本用于删除所有预处理生成的数据，包括：
- frames/ - 原始抽帧结果
- faces_aligned/ - 人脸对齐结果（缓存）
- features/ - 特征向量（缓存）
- clips/ - clip帧（最终训练样本）
- meta/ - 元数据
- dataset_index.pkl / dataset_index.npy - 数据集索引
- dataset_stats.json - 数据集统计信息

使用方法:
    python src/preprocess/clean_preprocessed_data.py
    或
    python src/preprocess/clean_preprocessed_data.py --output_root data/
"""
import os
import argparse
import shutil


def clean_preprocessed_data(output_root='data/', 
                            clean_frames=True,
                            clean_faces=True,
                            clean_features=True,
                            clean_clips=True,
                            clean_meta=True,
                            clean_index=True,
                            clean_stats=True,
                            confirm=True):
    """清理预处理生成的文件和目录
    
    Args:
        output_root: 预处理数据根目录（默认为 'data/'）
        clean_frames: 是否清理 frames 目录
        clean_faces: 是否清理 faces_aligned 目录
        clean_features: 是否清理 features 目录
        clean_clips: 是否清理 clips 目录
        clean_meta: 是否清理 meta 目录
        clean_index: 是否清理索引文件（.pkl 和 .npy）
        clean_stats: 是否清理统计信息文件（.json）
        confirm: 是否需要用户确认
    
    Returns:
        tuple: (删除的目录列表, 删除的文件列表)
    """
    output_root = os.path.abspath(output_root)
    
    if not os.path.exists(output_root):
        print(f"[WARN] 输出根目录不存在: {output_root}")
        return [], []
    
    dirs_to_remove = []
    files_to_remove = []
    
    # 要清理的目录
    dirs = {
        'frames': 'frames',
        'faces_aligned': 'faces_aligned',
        'features': 'features',
        'clips': 'clips',
        'meta': 'meta',
    }
    
    if clean_frames:
        dirs_to_remove.append(dirs['frames'])
    if clean_faces:
        dirs_to_remove.append(dirs['faces_aligned'])
    if clean_features:
        dirs_to_remove.append(dirs['features'])
    if clean_clips:
        dirs_to_remove.append(dirs['clips'])
    if clean_meta:
        dirs_to_remove.append(dirs['meta'])
    
    # 要清理的文件
    files = {
        'dataset_index.pkl': 'dataset_index.pkl',
        'dataset_index.npy': 'dataset_index.npy',
        'dataset_stats.json': 'dataset_stats.json',
    }
    
    if clean_index:
        files_to_remove.append(files['dataset_index.pkl'])
        files_to_remove.append(files['dataset_index.npy'])
    if clean_stats:
        files_to_remove.append(files['dataset_stats.json'])
    
    # 打印将要清理的内容
    print(f"\n{'='*60}")
    print("准备清理预处理数据")
    print(f"{'='*60}")
    print(f"输出根目录: {output_root}")
    print(f"\n将要删除的目录 ({len(dirs_to_remove)}):")
    for d in dirs_to_remove:
        path = os.path.join(output_root, d)
        if os.path.exists(path):
            size = _get_dir_size(path)
            print(f"  - {path} ({_format_size(size)})")
        else:
            print(f"  - {path} (不存在)")
    
    print(f"\n将要删除的文件 ({len(files_to_remove)}):")
    for f in files_to_remove:
        path = os.path.join(output_root, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  - {path} ({_format_size(size)})")
        else:
            print(f"  - {path} (不存在)")
    
    print(f"{'='*60}")
    
    # 确认
    if confirm:
        response = input("\n确认删除以上文件? (yes/no): ").strip().lower()
        if response not in ['yes', 'y', '是', '确认']:
            print("已取消")
            return [], []
    
    # 执行删除
    removed_dirs = []
    removed_files = []
    
    for d in dirs_to_remove:
        path = os.path.join(output_root, d)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                removed_dirs.append(path)
                print(f"[OK] 已删除目录: {path}")
            except Exception as e:
                print(f"[ERROR] 删除目录失败: {path}, 错误: {e}")
    
    for f in files_to_remove:
        path = os.path.join(output_root, f)
        if os.path.exists(path):
            try:
                os.remove(path)
                removed_files.append(path)
                print(f"[OK] 已删除文件: {path}")
            except Exception as e:
                print(f"[ERROR] 删除文件失败: {path}, 错误: {e}")
    
    print(f"\n{'='*60}")
    print("清理完成!")
    print(f"已删除目录: {len(removed_dirs)} 个")
    print(f"已删除文件: {len(removed_files)} 个")
    print(f"{'='*60}\n")
    
    return removed_dirs, removed_files


def _get_dir_size(path):
    """获取目录大小（字节）"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except Exception:
        pass
    return total


def _format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def main():
    parser = argparse.ArgumentParser(description='清理预处理生成的文件和目录')
    parser.add_argument('--output_root', type=str, default='data/',
                        help='预处理数据根目录（默认: data/）')
    parser.add_argument('--keep_frames', action='store_true',
                        help='保留 frames 目录')
    parser.add_argument('--keep_faces', action='store_true',
                        help='保留 faces_aligned 目录')
    parser.add_argument('--keep_features', action='store_true',
                        help='保留 features 目录')
    parser.add_argument('--keep_clips', action='store_true',
                        help='保留 clips 目录')
    parser.add_argument('--keep_meta', action='store_true',
                        help='保留 meta 目录')
    parser.add_argument('--keep_index', action='store_true',
                        help='保留索引文件')
    parser.add_argument('--keep_stats', action='store_true',
                        help='保留统计文件')
    parser.add_argument('--no_confirm', action='store_true',
                        help='不询问确认，直接删除')
    
    args = parser.parse_args()
    
    clean_preprocessed_data(
        output_root=args.output_root,
        clean_frames=not args.keep_frames,
        clean_faces=not args.keep_faces,
        clean_features=not args.keep_features,
        clean_clips=not args.keep_clips,
        clean_meta=not args.keep_meta,
        clean_index=not args.keep_index,
        clean_stats=not args.keep_stats,
        confirm=not args.no_confirm
    )


if __name__ == '__main__':
    main()

