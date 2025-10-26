"""人脸检测与五点对齐模块"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def align_face(img, detector=None, output_size=112):
    """检测并五点对齐人脸
    
    Args:
        img: 输入图像（BGR格式）
        detector: FaceAnalysis 实例（必传，不能为 None）
        output_size: 输出图像尺寸（默认 112）
    
    Returns:
        aligned: 对齐后的人脸图像，如果检测失败返回 None
    """
    if detector is None:
        raise ValueError("detector 不能为 None，请先调用 create_face_detector() 创建 detector")
    
    faces = detector.get(img)
    if len(faces) == 0:
        return None
    
    # 选择最大的人脸
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    landmarks = face.kps
    # 仿射变换
    ref = np.array([[30.2946, 51.6963], 
                    [65.5318, 51.5014],
                    [48.0252, 71.7366], 
                    [33.5493, 92.3655], 
                    [62.7299, 92.2041]], dtype=np.float32)
    if output_size == 112:
        dst = ref
    else:
        dst = ref * (output_size / 112)
    M = cv2.estimateAffinePartial2D(landmarks, dst, method=cv2.LMEDS)[0]
    aligned = cv2.warpAffine(img, M, (output_size, output_size))
    return aligned
