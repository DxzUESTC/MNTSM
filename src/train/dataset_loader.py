"""Dataset 加载模块"""
import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

# 降低多进程下的OpenCV线程竞争，提升整体DataLoader吞吐
try:
    cv2.setNumThreads(0)
except Exception:
    pass


class DeepfakeDataset(Dataset):
    def __init__(self, clip_list, data_root='data', transform=None, allow_skip=True, use_fast_io=True):
        """
        Args:
            clip_list: clip列表，每个元素包含clip_dir, frames, label等信息
            data_root: 数据根目录（默认 'data'）
            transform: 图像变换（可选）
            allow_skip: 是否允许跳过无效clip（默认True）
            use_fast_io: 是否启用更快的读取路径（np.fromfile + cv2.imdecode），在Windows上通常更稳更快
        """
        self.data_root = data_root
        self.transform = transform
        self.allow_skip = allow_skip
        self.use_fast_io = use_fast_io
        
        # 预过滤无效clips（可选，避免运行时错误）
        if allow_skip:
            self.clip_list = self._filter_valid_clips(clip_list)
        else:
            self.clip_list = clip_list
    
    def _filter_valid_clips(self, clip_list):
        """预过滤掉不存在的clip目录"""
        valid_clips = []
        skipped = 0
        for clip in clip_list:
            clip_dir = clip.get('clip_dir', '')
            if not clip_dir:
                skipped += 1
                continue
            
            clip_path = os.path.join(self.data_root, clip_dir)
            if not os.path.exists(clip_path):
                skipped += 1
                continue
            
            # 检查是否有帧文件
            frame_files = [f for f in os.listdir(clip_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(frame_files) == 0:
                skipped += 1
                continue
            
            valid_clips.append(clip)
        
        if skipped > 0:
            print(f"[WARN] 数据集加载时跳过了 {skipped} 个无效的 clips")
        
        return valid_clips

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        """
        加载clip的所有帧图像
        
        Returns:
            clip_tensor: (C, T, H, W) 格式的张量，C=通道数，T=帧数
            label: 标签 (0=真实, 1=伪造)
        """
        clip = self.clip_list[idx]
        clip_dir = clip['clip_dir']
        frames_info = clip.get('frames', [])
        label = clip.get('label', 0)
        
        clip_path = os.path.join(self.data_root, clip_dir)
        images = []
        
        # 加载所有帧
        for frame_info in frames_info:
            frame_name = frame_info.get('out_name', '')
            if not frame_name:
                continue
            
            frame_path = os.path.join(clip_path, frame_name)
            if not os.path.exists(frame_path):
                if self.allow_skip:
                    # 跳过缺失的帧
                    continue
                else:
                    raise FileNotFoundError(f"帧文件不存在: {frame_path}")
            
            try:
                if self.use_fast_io:
                    # 使用 fromfile + imdecode，兼容中文路径并减少IO开销
                    data = np.fromfile(frame_path, dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                else:
                    img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                if img is None:
                    if self.allow_skip:
                        continue
                    else:
                        raise ValueError(f"无法读取图像: {frame_path}")
                
                # BGR -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
            except Exception as e:
                if self.allow_skip:
                    continue
                else:
                    raise e
        
        # 如果所有帧都加载失败
        if len(images) == 0:
            if self.allow_skip:
                # 返回一个占位符（或者抛出异常，由DataLoader处理）
                # 这里返回全零张量作为占位符，实际训练时可能需要特殊处理
                images = [np.zeros((112, 112, 3), dtype=np.uint8)]
            else:
                raise ValueError(f"Clip {clip_dir} 没有可加载的帧")
        
        # 转换为numpy数组并堆叠
        images = np.stack(images, axis=0)  # (T, H, W, C)
        images = np.transpose(images, (3, 0, 1, 2))  # (C, T, H, W)
        
        # 转换为tensor
        clip_tensor = torch.from_numpy(images).float() / 255.0
        
        # 应用变换（如果有）
        if self.transform is not None:
            # transform应该处理 (C, T, H, W) 格式
            clip_tensor = self.transform(clip_tensor)
        
        # 组装用于 video-level 聚合的元信息（评估阶段会使用）
        meta = {
            'video': clip.get('video', ''),
            'raw_rel_path': clip.get('raw_rel_path', ''),
            'clip_dir': clip_dir,
            'dataset_name': clip.get('dataset_name', 'Unknown'),
        }

        return clip_tensor, label, meta
