"""人脸检测与五点对齐模块"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def align_face(img, detector=None, output_size=112, expand_ratio=0.2):
    """检测并五点对齐人脸
    
    Args:
        img: 输入图像（BGR格式）
        detector: FaceAnalysis 实例（必传，不能为 None）
        output_size: 输出图像尺寸（默认 112）
        expand_ratio: 外扩比例（默认0.2即20%），确保人脸完整（包括额头、下巴和侧边）
    
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
    
    # 原始参考点（五点对齐的标准位置）
    ref = np.array([[30.2946, 51.6963], 
                    [65.5318, 51.5014],
                    [48.0252, 71.7366], 
                    [33.5493, 92.3655], 
                    [62.7299, 92.2041]], dtype=np.float32)
    
    # 根据output_size调整参考点
    if output_size != 112:
        ref = ref * (output_size / 112)
    
    # 计算参考中心点（鼻子位置）
    ref_center = ref[2]  # 第三个点是鼻子
    
    # 外扩：将参考点以中心为基准缩小（相当于扩大裁剪范围）
    # 例如 expand_ratio=0.4 时，scale=1/1.4≈0.71，意味着会在更大的范围内进行对齐（相当于外扩40%）
    # 这样可以包含更多的额头、下巴和侧边区域，避免裁剪到人脸
    scale = 1.0 / (1.0 + expand_ratio)
    dst = ref_center + (ref - ref_center) * scale
    
    M = cv2.estimateAffinePartial2D(landmarks, dst, method=cv2.LMEDS)[0]
    aligned = cv2.warpAffine(img, M, (output_size, output_size))
    return aligned