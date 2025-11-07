"""验证Celeb-DF-v2数据划分的完整性

检查：
1. 视频级泄露：同一视频的不同clips是否都在同一集合
2. ID级泄露：同一ID的真实和伪造视频是否都在同一集合
3. 帧级泄露：同一视频的不同帧是否都在同一集合
"""
import pickle
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.dataset_split import extract_celebdf_video_id


def verify_split(split_path='data/splits/Celeb-DF-v2_split_42.pkl'):
    """验证数据划分的完整性"""
    print("=" * 80)
    print("Celeb-DF-v2 数据划分完整性验证")
    print("=" * 80)
    
    # 加载划分数据
    with open(split_path, 'rb') as f:
        data = pickle.load(f)
    
    train_clips = data['train_clips']
    val_clips = data['val_clips']
    test_clips = data['test_clips']
    
    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_clips)} clips")
    print(f"  验证集: {len(val_clips)} clips")
    print(f"  测试集: {len(test_clips)} clips")
    
    # 1. 视频级检查：同一视频的所有clips是否都在同一集合
    print("\n" + "=" * 80)
    print("1. 视频级泄露检查（同一视频的不同clips是否都在同一集合）")
    print("=" * 80)
    
    train_videos = {}
    val_videos = {}
    test_videos = {}
    
    for clip in train_clips:
        video_id = clip.get('raw_rel_path', '')
        if video_id:
            train_videos.setdefault(video_id, []).append(clip)
    
    for clip in val_clips:
        video_id = clip.get('raw_rel_path', '')
        if video_id:
            val_videos.setdefault(video_id, []).append(clip)
    
    for clip in test_clips:
        video_id = clip.get('raw_rel_path', '')
        if video_id:
            test_videos.setdefault(video_id, []).append(clip)
    
    train_video_set = set(train_videos.keys())
    val_video_set = set(val_videos.keys())
    test_video_set = set(test_videos.keys())
    
    train_val_overlap = train_video_set & val_video_set
    train_test_overlap = train_video_set & test_video_set
    val_test_overlap = val_video_set & test_video_set
    
    print(f"  训练集视频数: {len(train_video_set)}")
    print(f"  验证集视频数: {len(val_video_set)}")
    print(f"  测试集视频数: {len(test_video_set)}")
    print(f"  训练-验证重叠: {len(train_val_overlap)} 个视频")
    print(f"  训练-测试重叠: {len(train_test_overlap)} 个视频")
    print(f"  验证-测试重叠: {len(val_test_overlap)} 个视频")
    
    if train_val_overlap:
        print(f"  ⚠️ 发现训练-验证重叠视频: {list(train_val_overlap)[:5]}")
    if train_test_overlap:
        print(f"  ⚠️ 发现训练-测试重叠视频: {list(train_test_overlap)[:5]}")
    if val_test_overlap:
        print(f"  ⚠️ 发现验证-测试重叠视频: {list(val_test_overlap)[:5]}")
    
    video_leak = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    if video_leak == 0:
        print("  ✓ 视频级划分正确，无泄露")
    else:
        print(f"  ✗ 发现 {video_leak} 个视频存在泄露")
    
    # 2. ID级检查：同一ID的真实和伪造视频是否都在同一集合
    print("\n" + "=" * 80)
    print("2. ID级泄露检查（同一ID的真实和伪造视频是否都在同一集合）")
    print("=" * 80)
    
    train_ids = {}
    val_ids = {}
    test_ids = {}
    
    for clip in train_clips:
        video_id = clip.get('raw_rel_path', '')
        celebdf_id = extract_celebdf_video_id(video_id)
        if celebdf_id:
            train_ids.setdefault(celebdf_id, []).append(clip)
    
    for clip in val_clips:
        video_id = clip.get('raw_rel_path', '')
        celebdf_id = extract_celebdf_video_id(video_id)
        if celebdf_id:
            val_ids.setdefault(celebdf_id, []).append(clip)
    
    for clip in test_clips:
        video_id = clip.get('raw_rel_path', '')
        celebdf_id = extract_celebdf_video_id(video_id)
        if celebdf_id:
            test_ids.setdefault(celebdf_id, []).append(clip)
    
    train_id_set = set(train_ids.keys())
    val_id_set = set(val_ids.keys())
    test_id_set = set(test_ids.keys())
    
    train_val_id_overlap = train_id_set & val_id_set
    train_test_id_overlap = train_id_set & test_id_set
    val_test_id_overlap = val_id_set & test_id_set
    
    print(f"  训练集ID数: {len(train_id_set)}")
    print(f"  验证集ID数: {len(val_id_set)}")
    print(f"  测试集ID数: {len(test_id_set)}")
    print(f"  训练-验证重叠ID: {len(train_val_id_overlap)} 个")
    print(f"  训练-测试重叠ID: {len(train_test_id_overlap)} 个")
    print(f"  验证-测试重叠ID: {len(val_test_id_overlap)} 个")
    
    if train_val_id_overlap:
        print(f"  ⚠️ 发现训练-验证重叠ID: {list(train_val_id_overlap)[:5]}")
        for id in list(train_val_id_overlap)[:3]:
            train_clips_for_id = train_ids[id]
            val_clips_for_id = val_ids[id]
            train_labels = set(c.get('label', 0) for c in train_clips_for_id)
            val_labels = set(c.get('label', 0) for c in val_clips_for_id)
            print(f"    ID {id}: 训练集{len(train_clips_for_id)}个clips(标签:{train_labels}), 验证集{len(val_clips_for_id)}个clips(标签:{val_labels})")
    
    if train_test_id_overlap:
        print(f"  ⚠️ 发现训练-测试重叠ID: {list(train_test_id_overlap)[:5]}")
        for id in list(train_test_id_overlap)[:3]:
            train_clips_for_id = train_ids[id]
            test_clips_for_id = test_ids[id]
            train_labels = set(c.get('label', 0) for c in train_clips_for_id)
            test_labels = set(c.get('label', 0) for c in test_clips_for_id)
            print(f"    ID {id}: 训练集{len(train_clips_for_id)}个clips(标签:{train_labels}), 测试集{len(test_clips_for_id)}个clips(标签:{test_labels})")
    
    if val_test_id_overlap:
        print(f"  ⚠️ 发现验证-测试重叠ID: {list(val_test_id_overlap)[:5]}")
    
    id_leak = len(train_val_id_overlap) + len(train_test_id_overlap) + len(val_test_id_overlap)
    if id_leak == 0:
        print("  ✓ ID级划分正确，无泄露")
    else:
        print(f"  ✗ 发现 {id_leak} 个ID存在泄露")
    
    # 3. 帧级检查：同一视频的不同帧是否都在同一集合
    print("\n" + "=" * 80)
    print("3. 帧级泄露检查（同一视频的不同帧是否都在同一集合）")
    print("=" * 80)
    
    train_frames = {}
    val_frames = {}
    test_frames = {}
    
    for clip in train_clips:
        video_id = clip.get('raw_rel_path', '')
        for frame_info in clip.get('frames', []):
            frame_key = (video_id, frame_info.get('out_name', ''))
            if frame_key[0] and frame_key[1]:
                train_frames[frame_key] = clip
    
    for clip in val_clips:
        video_id = clip.get('raw_rel_path', '')
        for frame_info in clip.get('frames', []):
            frame_key = (video_id, frame_info.get('out_name', ''))
            if frame_key[0] and frame_key[1]:
                val_frames[frame_key] = clip
    
    for clip in test_clips:
        video_id = clip.get('raw_rel_path', '')
        for frame_info in clip.get('frames', []):
            frame_key = (video_id, frame_info.get('out_name', ''))
            if frame_key[0] and frame_key[1]:
                test_frames[frame_key] = clip
    
    train_frame_set = set(train_frames.keys())
    val_frame_set = set(val_frames.keys())
    test_frame_set = set(test_frames.keys())
    
    train_val_frame_overlap = train_frame_set & val_frame_set
    train_test_frame_overlap = train_frame_set & test_frame_set
    val_test_frame_overlap = val_frame_set & test_frame_set
    
    print(f"  训练集帧数: {len(train_frame_set)}")
    print(f"  验证集帧数: {len(val_frame_set)}")
    print(f"  测试集帧数: {len(test_frame_set)}")
    print(f"  训练-验证重叠帧: {len(train_val_frame_overlap)} 个")
    print(f"  训练-测试重叠帧: {len(train_test_frame_overlap)} 个")
    print(f"  验证-测试重叠帧: {len(val_test_frame_overlap)} 个")
    
    if train_val_frame_overlap:
        print(f"  ⚠️ 发现训练-验证重叠帧: {list(train_val_frame_overlap)[:3]}")
    if train_test_frame_overlap:
        print(f"  ⚠️ 发现训练-测试重叠帧: {list(train_test_frame_overlap)[:3]}")
    if val_test_frame_overlap:
        print(f"  ⚠️ 发现验证-测试重叠帧: {list(val_test_frame_overlap)[:3]}")
    
    frame_leak = len(train_val_frame_overlap) + len(train_test_frame_overlap) + len(val_test_frame_overlap)
    if frame_leak == 0:
        print("  ✓ 帧级划分正确，无泄露")
    else:
        print(f"  ✗ 发现 {frame_leak} 个帧存在泄露")
    
    # 总结
    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)
    
    all_ok = (video_leak == 0) and (id_leak == 0) and (frame_leak == 0)
    
    if all_ok:
        print("✓ 所有检查通过，数据划分正确，无泄露")
    else:
        print("✗ 发现数据泄露问题:")
        if video_leak > 0:
            print(f"  - 视频级泄露: {video_leak} 个视频")
        if id_leak > 0:
            print(f"  - ID级泄露: {id_leak} 个ID")
        if frame_leak > 0:
            print(f"  - 帧级泄露: {frame_leak} 个帧")
    
    return all_ok


if __name__ == '__main__':
    verify_split()

