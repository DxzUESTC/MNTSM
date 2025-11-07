"""检查评估实现的关键点

验证我们的评估方式是否符合标准实践
"""
import pickle
from statistics import mean, median

def check_evaluation_implementation():
    """检查评估实现的关键点"""
    print("=" * 80)
    print("评估实现检查")
    print("=" * 80)
    
    # 加载测试集数据
    with open('data/splits/Celeb-DF-v2_split_42.pkl', 'rb') as f:
        data = pickle.load(f)
    
    test_clips = data['test_clips']
    
    # 1. 检查测试集统计
    print("\n1. 测试集统计:")
    test_frames = sum(len(c.get('frames', [])) for c in test_clips)
    test_videos = len(set(c.get('raw_rel_path', '') for c in test_clips))
    test_real = sum(1 for c in test_clips if c.get('label', 0) == 0)
    test_fake = sum(1 for c in test_clips if c.get('label', 0) == 1)
    
    print(f"  视频数: {test_videos}")
    print(f"  Clips数: {len(test_clips)}")
    print(f"  帧数: {test_frames}")
    print(f"  真实clips: {test_real} ({test_real/len(test_clips)*100:.1f}%)")
    print(f"  伪造clips: {test_fake} ({test_fake/len(test_clips)*100:.1f}%)")
    
    # 2. 检查帧分布
    print("\n2. 帧分布检查:")
    frames_per_video = []
    for video_id in set(c.get('raw_rel_path', '') for c in test_clips):
        video_clips = [c for c in test_clips if c.get('raw_rel_path', '') == video_id]
        video_frames = sum(len(c.get('frames', [])) for c in video_clips)
        frames_per_video.append(video_frames)
    
    print(f"  每视频平均帧数: {mean(frames_per_video):.1f}")
    print(f"  每视频帧数范围: {min(frames_per_video)} - {max(frames_per_video)}")
    print(f"  每视频帧数中位数: {median(frames_per_video):.1f}")
    
    # 3. 检查AUC计算方式
    print("\n3. AUC计算方式检查:")
    print("  我们使用: sklearn.metrics.roc_auc_score")
    print("  这是标准的AUC计算方法")
    
    # 4. 模拟评估过程
    print("\n4. 评估过程模拟:")
    print("  Frame-level AUC:")
    print("    - 收集所有帧的预测概率和标签")
    print("    - 直接计算AUC: roc_auc_score(all_frame_labels, all_frame_probs)")
    print("    - 没有采样或过滤")
    print("    - 所有帧都参与计算")
    
    print("\n  Clip-level AUC:")
    print("    - 将同一clip的所有帧概率聚合（mean/max/attention）")
    print("    - 每个clip得到一个概率")
    print("    - 计算clip-level AUC")
    
    print("\n  Video-level AUC:")
    print("    - 将同一video的所有clip概率做平均")
    print("    - 每个video得到一个概率")
    print("    - 计算video-level AUC")
    
    # 5. 可能的差异点
    print("\n5. 与SOTA可能的差异点:")
    print("  a) 帧采样:")
    print("     - 我们: 使用所有帧")
    print("     - SOTA可能: 每视频采样固定数量帧")
    print("  b) 数据预处理:")
    print("     - 我们: 人脸对齐 + 224x224裁剪")
    print("     - SOTA可能: 不同的预处理流程")
    print("  c) 评估指标:")
    print("     - 我们: 使用sklearn标准实现")
    print("     - SOTA: 可能相同，但样本可能不同")
    
    # 6. 建议的验证步骤
    print("\n6. 建议的验证步骤:")
    print("  1. 查找FreqBlender和DefakeHop论文中的评估细节")
    print("  2. 检查是否有'每视频采样N帧'的说明")
    print("  3. 尝试不同的帧采样策略")
    print("  4. 对比不同预处理方式的结果")
    print("  5. 检查数据集版本是否一致")
    
    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)

if __name__ == '__main__':
    check_evaluation_implementation()

