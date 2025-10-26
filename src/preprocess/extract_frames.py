"""视频抽帧模块"""
import cv2
import os

def extract_frames(video_path: str, output_dir: str, sample_fps: int = 4):
    """从视频中按指定帧率抽帧并保存

    Args:
        video_path: 输入视频路径
        output_dir: 输出帧保存目录
        sample_fps: 抽帧帧率 (默认 4fps)
    
    Returns:
        int: 成功保存的帧数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(video_path):
        print(f"[ERROR] 视频文件不存在: {video_path}")
        return 0
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频文件: {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"[WARN] 无法获取视频FPS，使用默认值 25.0")
        fps = 25.0
    
    # 确保 interval 至少为 1
    interval = max(1, int(fps / sample_fps))
    frame_idx = 0
    saved = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                save_path = os.path.join(output_dir, f"{saved:06d}.jpg")
                success = cv2.imwrite(save_path, frame)
                if success:
                    saved += 1
            frame_idx += 1
    finally:
        cap.release()
    
    return saved
