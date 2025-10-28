"""clip 生成模块"""
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from extract_frames import extract_frames
from face_align import align_face
from keyframe_selection import select_keyframes
from insightface.app import FaceAnalysis


def create_face_detector(use_gpu=True):
    """创建并初始化人脸检测器（建议在主进程中调用一次后复用）
    
    Args:
        use_gpu: 是否使用 GPU (默认 True，会尝试 CUDA，失败则回退到 CPU)
    Returns:
        detector: 已准备好的 FaceAnalysis 实例
    """
    if use_gpu:
        try:
            detector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            detector.prepare(ctx_id=0, det_size=(640, 640))
            print("[INFO] FaceAnalysis initialized with CUDA")
        except Exception as e:
            print(f"[WARN] CUDA init failed: {e}, fallback to CPU")
            detector = FaceAnalysis(providers=['CPUExecutionProvider'])
            detector.prepare(ctx_id=-1, det_size=(640, 640))
    else:
        detector = FaceAnalysis(providers=['CPUExecutionProvider'])
        detector.prepare(ctx_id=-1, det_size=(640, 640))
        print("[INFO] FaceAnalysis initialized with CPU")
    return detector


def compute_face_feature(aligned_face):
    """计算对齐人脸的局部区域 L1 差分特征（替代简单灰度降采样）
    
    将 112x112 对齐人脸划分为 4x4 区域，计算每个区域的均值和方差，再计算相邻区域间的 L1 差分。
    返回 32 维特征向量（16 个区域均值 + 16 个方差）。
    """
    gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    grid_size = 4
    cell_h, cell_w = h // grid_size, w // grid_size
    
    features = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            features.append(cell.mean())
            features.append(cell.std())
    
    return np.array(features, dtype=np.float32)


def is_video_processed(video_path, output_root, raw_root=None):
    """检查视频是否已经被成功处理过
    
    Args:
        video_path: 原始视频文件完整路径
        output_root: 存放处理结果的根目录
        raw_root: raw_videos 的根目录
    
    Returns:
        bool: True表示已处理且完整，False表示未处理或不完整
    """
    # 计算相对路径
    if raw_root:
        rel_path = os.path.relpath(video_path, raw_root)
    else:
        rel_path = os.path.basename(video_path)
    rel_dir = os.path.dirname(rel_path)
    video_stem = os.path.splitext(os.path.basename(rel_path))[0]
    
    meta_dir = os.path.join(output_root, 'meta', rel_dir, video_stem)
    meta_path = os.path.join(meta_dir, 'clip_meta.json')
    
    # 检查 meta 文件是否存在且有效
    if not os.path.exists(meta_path):
        return False
    
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        # 检查是否有错误标记
        if 'error' in meta:
            return False
        
        # 检查是否有有效的 clips
        if len(meta.get('clips', [])) == 0:
            return False
        
        # 检查 clips 目录是否存在且不为空
        clip_dir = os.path.join(output_root, 'clips', rel_dir, video_stem)
        if not os.path.exists(clip_dir):
            return False
        
        # 检查至少有一个 clip 目录存在
        clip_subdirs = [d for d in os.listdir(clip_dir) if d.startswith('clip_')]
        if len(clip_subdirs) == 0:
            return False
        
        return True
    except Exception:
        return False


def build_clips(video_path, output_root, detector, fps=4, clip_len=8, raw_root=None, stride=None, 
                skip_if_processed=False):
    """
    从视频生成可训练的 clip（抽帧 -> 对齐 -> 关键帧选择 -> 滑窗生成多 clip）
    输出会镜像 raw_root 下的视频相对路径。
    
    改进点：
    1. detector 由外部传入（避免重复初始化）
    2. 使用改进的人脸局部区域 L1 差分特征
    3. 当抽帧总数 > 32 时，使用滑窗生成多个 clip

    参数:
      - video_path: 原始视频文件完整路径
      - output_root: 存放处理结果的根目录（例如 data/）
      - detector: 已初始化的 FaceAnalysis 实例（必须传入）
      - fps: 抽帧帧率
      - clip_len: 每个 clip 的帧数
      - raw_root: raw_videos 的根目录，用于保持目录结构（例如 data/raw_videos）
      - stride: 滑窗步长（默认为 clip_len // 2，即 50% 重叠）
      - skip_if_processed: 如果视频已处理过是否跳过
    返回:
      - (meta dict, status): meta信息和状态('processed'/'skipped'/'failed')
    """
    if detector is None:
        raise ValueError("detector 不能为 None，请先调用 create_face_detector() 创建并传入")
    
    if stride is None:
        stride = max(1, clip_len // 2)
    
    # 计算相对路径（用于镜像目录结构）
    if raw_root:
        rel_path = os.path.relpath(video_path, raw_root)
    else:
        rel_path = os.path.basename(video_path)
    rel_dir = os.path.dirname(rel_path)  # 可能为空
    video_stem = os.path.splitext(os.path.basename(rel_path))[0]

    # 构建各功能目录，镜像 raw_videos 结构： e.g. output_root/frames/<rel_dir>/<video_stem>/
    frame_dir = os.path.join(output_root, 'frames', rel_dir, video_stem)
    aligned_dir = os.path.join(output_root, 'faces_aligned', rel_dir, video_stem)
    feature_dir = os.path.join(output_root, 'features', rel_dir, video_stem)
    clip_dir = os.path.join(output_root, 'clips', rel_dir, video_stem)
    meta_dir = os.path.join(output_root, 'meta', rel_dir, video_stem)
    
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    
    # 如果设置了跳过已处理的视频且该视频已处理过，直接返回
    if skip_if_processed and is_video_processed(video_path, output_root, raw_root):
        try:
            meta_path = os.path.join(meta_dir, 'clip_meta.json')
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            return meta, 'skipped'
        except Exception:
            pass

    # 1）抽帧（会在 frame_dir 下写入帧图片）
    num_extracted = extract_frames(video_path, frame_dir, fps)
    
    if num_extracted == 0:
        print(f"[WARN] 未能从视频抽帧: {video_path}")
        meta = {"video": video_path, "clip_len": clip_len, "clips": []}
        json.dump(meta, open(os.path.join(meta_dir, 'clip_meta.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        return meta, 'failed'

    try:
        # 2) 读取抽取帧并对齐，生成特征向量列表
        frame_files = sorted([f for f in os.listdir(frame_dir)
                              if f.lower().endswith(('.jpg', '.png'))])
        if len(frame_files) == 0:
            meta = {"video": video_path, "clip_len": clip_len, "clips": []}
            json.dump(meta, open(os.path.join(meta_dir, 'clip_meta.json'), 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=2)
            return meta, 'failed'

        features = []
        aligned_imgs = []
        src_indices = []  # 原始帧索引对应关系（相对于 frame_files 列表）
        
        # 添加进度条显示对齐人脸的处理进度
        for i, fname in enumerate(tqdm(frame_files, desc="人脸对齐", leave=False)):
            # 构建缓存文件路径
            aligned_cache_path = os.path.join(aligned_dir, fname)
            # 将 .jpg/.png 替换为 .npy
            feature_fname = fname.replace('.jpg', '.npy').replace('.png', '.npy')
            feature_cache_path = os.path.join(feature_dir, feature_fname)
            
            aligned = None
            feat = None
            
            # 先尝试从缓存加载
            if os.path.exists(aligned_cache_path) and os.path.exists(feature_cache_path):
                try:
                    aligned = cv2.imread(aligned_cache_path)
                    feat = np.load(feature_cache_path)
                    if aligned is not None and feat is not None:
                        aligned_imgs.append(aligned)
                        features.append(feat)
                        src_indices.append(i)
                        continue
                except Exception as e:
                    print(f"[WARN] 加载缓存失败，将重新处理: {fname}, 错误: {e}")
            
            # 如果缓存不存在或加载失败，重新处理
            fp = os.path.join(frame_dir, fname)
            img = cv2.imread(fp)
            if img is None:
                continue
            aligned = align_face(img, detector=detector, output_size=112)
            if aligned is None:
                # 若检测/对齐失败，跳过该帧
                continue
            
            # 保存对齐图像到缓存
            try:
                cv2.imwrite(aligned_cache_path, aligned)
            except Exception as e:
                print(f"[WARN] 保存对齐图像缓存失败: {aligned_cache_path}, 错误: {e}")
            
            # 使用改进的局部区域特征（均值+方差）
            feat = compute_face_feature(aligned)
            
            # 保存特征到缓存
            try:
                np.save(feature_cache_path, feat)
            except Exception as e:
                print(f"[WARN] 保存特征缓存失败: {feature_cache_path}, 错误: {e}")
            
            features.append(feat)
            aligned_imgs.append(aligned)
            src_indices.append(i)

        if len(features) == 0:
            meta = {"video": video_path, "clip_len": clip_len, "clips": []}
            json.dump(meta, open(os.path.join(meta_dir, 'clip_meta.json'), 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=2)
            return meta, 'failed'

        # 3) 滑窗生成多个 clip（当帧数 > 32 时）
        num_aligned = len(features)
        clips_info = []
        
        if num_aligned <= 32:
            # 短视频：直接用 select_keyframes 选一个 clip
            selected_feat_idx = select_keyframes(features, clip_len=clip_len)
            clip_id = 0
            clip_subdir = os.path.join(clip_dir, f"clip_{clip_id:03d}")
            os.makedirs(clip_subdir, exist_ok=True)
            
            selected_frames = []
            for out_i, feat_i in enumerate(selected_feat_idx):
                feat_i = int(feat_i)
                orig_frame_idx = src_indices[feat_i]
                fname = frame_files[orig_frame_idx]
                dst_name = f"{out_i:06d}.jpg"
                dst_path = os.path.join(clip_subdir, dst_name)
                aligned_img = aligned_imgs[feat_i]
                cv2.imwrite(dst_path, aligned_img)
                selected_frames.append({
                    "out_name": dst_name, 
                    "src_name": fname, 
                    "src_index": int(orig_frame_idx),
                    "feat_index": feat_i
                })
            
            clips_info.append({
                "clip_id": clip_id,
                "clip_dir": os.path.relpath(clip_subdir, output_root),
                "frames": selected_frames
            })
        else:
            # 长视频：滑窗生成多个 clip
            clip_id = 0
            for start_idx in range(0, num_aligned - clip_len + 1, stride):
                end_idx = start_idx + clip_len
                # 取当前窗口的特征子集
                window_features = features[start_idx:end_idx]
                # 在窗口内再做一次关键帧选择（可选，也可以直接取所有帧）
                # 这里简化为直接取窗口内所有帧（已经是 clip_len 长度）
                
                clip_subdir = os.path.join(clip_dir, f"clip_{clip_id:03d}")
                os.makedirs(clip_subdir, exist_ok=True)
                
                selected_frames = []
                for out_i, feat_i in enumerate(range(start_idx, end_idx)):
                    orig_frame_idx = src_indices[feat_i]
                    fname = frame_files[orig_frame_idx]
                    dst_name = f"{out_i:06d}.jpg"
                    dst_path = os.path.join(clip_subdir, dst_name)
                    aligned_img = aligned_imgs[feat_i]
                    cv2.imwrite(dst_path, aligned_img)
                    selected_frames.append({
                        "out_name": dst_name,
                        "src_name": fname,
                        "src_index": int(orig_frame_idx),
                        "feat_index": feat_i
                    })
                
                clips_info.append({
                    "clip_id": clip_id,
                    "clip_dir": os.path.relpath(clip_subdir, output_root),
                    "start_feat_idx": start_idx,
                    "end_feat_idx": end_idx,
                    "frames": selected_frames
                })
                clip_id += 1

        # 4) 保存元信息到 meta_dir
        meta = {
            "video": video_path,
            "raw_rel_path": rel_path,
            "fps": fps,
            "clip_len": clip_len,
            "stride": stride,
            "num_extracted_frames": len(frame_files),
            "num_aligned_frames": num_aligned,
            "num_clips": len(clips_info),
            "clips": clips_info,
            "frame_dir": os.path.relpath(frame_dir, output_root),
        }
        
        meta_path = os.path.join(meta_dir, 'clip_meta.json')
        json.dump(meta, open(meta_path, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        
        return meta, 'processed'
        
    except Exception as e:
        print(f"[ERROR] 处理视频失败 {video_path}: {e}")
        # 保存错误信息到 meta
        error_meta = {
            "video": video_path,
            "error": str(e),
            "clips": []
        }
        meta_path = os.path.join(meta_dir, 'clip_meta.json')
        try:
            json.dump(error_meta, open(meta_path, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=2)
            return error_meta, 'failed'
        except:
            pass
        return None, 'failed'
