"""模型训练模块"""
import os
import random
import pickle
import math
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F

from .dataset_loader import DeepfakeDataset
from .evaluator import evaluate
from ..models.mobilenetv4_tsm import create_mntsm_model
from ..utils.logger import experiment_logger


def _aggregate_time_dimension(outputs, batch_size, num_segments, mode="mean"):
    """将 (B*T, ...) 的模型输出还原为 (B, T, ...) 并按时间聚合为 (B, ...)。"""
    if outputs.dim() == 1:
        outputs = outputs.view(batch_size, num_segments)
        if mode == "mean":
            return outputs.mean(dim=1)
        elif mode == "max":
            return outputs.max(dim=1).values
        elif mode == "attention":
            # 非参数化注意力：对每帧logit做softmax权重
            scores = outputs  # (B, T)
            alpha = torch.softmax(scores, dim=1)
            probs = torch.sigmoid(scores)
            return (alpha * probs).sum(dim=1)
        else:
            return outputs.mean(dim=1)
    else:
        feature_dim = outputs.size(-1)
        outputs = outputs.view(batch_size, num_segments, feature_dim)
        if mode == "mean":
            return outputs.mean(dim=1)
        elif mode == "max":
            return outputs.max(dim=1).values
        elif mode == "attention":
            # 使用最后一维作为打分向量的投影（无参数近似）：score=||F_t||1
            scores = outputs.abs().sum(dim=-1, keepdim=True)  # (B,T,1)
            alpha = torch.softmax(scores, dim=1)
            return (alpha * outputs).sum(dim=1)
        else:
            return outputs.mean(dim=1)


def _prepare_targets_for_criterion(criterion, targets, outputs):
    """根据损失函数/输出形状，准备标签张量的数据类型与形状。"""
    # BCE 家族: 单通道或与输出同形状，float 标签 (0./1.)
    if isinstance(criterion, (nn.BCEWithLogitsLoss, nn.BCELoss)) or outputs.size(-1) == 1:
        if targets.dtype != torch.float32:
            targets = targets.float()
        if outputs.dim() == 2 and outputs.size(-1) == 1:
            targets = targets.view(-1, 1)
        return targets
    # CE 家族: 类别索引，long 标签
    if targets.dtype != torch.long:
        targets = targets.long()
    return targets


def train_one_epoch(model, dataloader, criterion, optimizer, device, n_segment=None, aggregate="mean", amp=False, scaler=None, grad_accum_steps: int = 1):
    """单轮训练。

    Args:
        model: 带有 TSM 的模型（期望输入形状为 (B*T, C, H, W)）。
        dataloader: 产生 (clip_tensor, label) 的迭代器，clip_tensor 形状为 (B, C, T, H, W)。
        criterion: 损失函数（CrossEntropyLoss 或 BCEWithLogitsLoss 等）。
        optimizer: 优化器。
        device: torch.device。
        n_segment: 时序片段数 T；若为 None，自动从 batch 推断。
        aggregate: 时序聚合方式（"mean" 或 "max"）。

    Returns:
        平均训练损失 (float)。
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    step_in_accum = 0
    for batch in dataloader:
        # 支持 (clip_tensor, label) 或 (clip_tensor, label, meta)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            clip_tensor, labels, _ = batch
        else:
            clip_tensor, labels = batch
        # clip_tensor: (B, C, T, H, W)
        clip_tensor = clip_tensor.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        B, C, T, H, W = clip_tensor.shape
        T_eff = T if n_segment is None else n_segment
        if T != T_eff:
            # 若给定 n_segment 与实际 T 不一致，以实际 T 为准并给予提示（一次性，可在外层日志控制）
            T_eff = T

        # (B, C, T, H, W) -> (B, T, C, H, W) -> (B*T, C, H, W)
        inputs = clip_tensor.permute(0, 2, 1, 3, 4).contiguous().view(B * T_eff, C, H, W)

        if step_in_accum == 0:
            optimizer.zero_grad(set_to_none=True)
        if amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                outputs = _aggregate_time_dimension(outputs, batch_size=B, num_segments=T_eff, mode=aggregate)
                labels_prepared = _prepare_targets_for_criterion(criterion, labels, outputs)
                loss = criterion(outputs, labels_prepared)
            loss = loss / max(1, grad_accum_steps)
            scaler.scale(loss).backward()
            step_in_accum += 1
            if step_in_accum % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                step_in_accum = 0
            total_loss += loss.detach().item() * max(1, grad_accum_steps)
            num_batches += 1
            continue
        else:
            outputs = model(inputs)

        # 将输出还原到 (B, ...) 做时序聚合
        outputs = _aggregate_time_dimension(outputs, batch_size=B, num_segments=T_eff, mode=aggregate)

        # 适配不同损失（CE / BCE）
        labels_prepared = _prepare_targets_for_criterion(criterion, labels, outputs)
        loss = criterion(outputs, labels_prepared)

        loss = loss / max(1, grad_accum_steps)
        loss.backward()
        step_in_accum += 1
        if step_in_accum % grad_accum_steps == 0:
            optimizer.step()
            step_in_accum = 0

        total_loss += loss.detach().item()
        num_batches += 1

    # 若剩余未step的累积梯度，在AMP关闭下已无；在AMP下也已覆盖
    return total_loss / max(1, num_batches)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_index(index_path):
    with open(index_path, 'rb') as f:
        data = pickle.load(f)
    # 兼容 {'clips': [...]} 或直接 list
    return data['clips'] if isinstance(data, dict) and 'clips' in data else data


def _split_clips_three(clips, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    按视频进行 训练/验证/测试 分层随机划分，确保同一视频的所有clips都在同一个集合中。
    
    这避免了数据泄露：同一视频的不同clips如果在训练/验证/测试集中分布，会导致过拟合。
    """
    assert val_ratio >= 0 and test_ratio >= 0 and (val_ratio + test_ratio) < 1.0
    
    # 按视频分组：使用 raw_rel_path 作为视频的唯一标识
    video_to_clips = {}
    for clip in clips:
        video_id = clip.get('raw_rel_path', '')
        if video_id not in video_to_clips:
            video_to_clips[video_id] = []
        video_to_clips[video_id].append(clip)
    
    # 将视频按真实/伪造分类
    real_videos = []  # 每个元素是 (video_id, clips_list, label)
    fake_videos = []
    for video_id, video_clips in video_to_clips.items():
        # 使用第一个clip的标签（同一个视频的所有clips标签应该一致）
        label = video_clips[0].get('label', 0)
        if label == 0:
            real_videos.append((video_id, video_clips, label))
        else:
            fake_videos.append((video_id, video_clips, label))
    
    # 随机打乱视频列表
    rng = random.Random(seed)
    rng.shuffle(real_videos)
    rng.shuffle(fake_videos)
    
    def split_three(lst):
        """按视频划分"""
        n_total = len(lst)
        n_val = int(math.floor(n_total * val_ratio))
        n_test = int(math.floor(n_total * test_ratio))
        n_train = max(0, n_total - n_val - n_test)
        train_part = lst[:n_train]
        val_part = lst[n_train:n_train+n_val]
        test_part = lst[n_train+n_val:n_train+n_val+n_test]
        return train_part, val_part, test_part
    
    # 分别对真实和伪造视频进行划分
    real_tr, real_va, real_te = split_three(real_videos)
    fake_tr, fake_va, fake_te = split_three(fake_videos)
    
    # 将视频列表展平为clips列表
    def flatten_videos(video_list):
        clips_list = []
        for _, video_clips, _ in video_list:
            clips_list.extend(video_clips)
        return clips_list
    
    train_clips = flatten_videos(real_tr) + flatten_videos(fake_tr)
    val_clips = flatten_videos(real_va) + flatten_videos(fake_va)
    test_clips = flatten_videos(real_te) + flatten_videos(fake_te)
    
    # 最后打乱clips顺序（但保持视频级划分不变）
    rng.shuffle(train_clips)
    rng.shuffle(val_clips)
    rng.shuffle(test_clips)
    
    return train_clips, val_clips, test_clips


def _compute_class_counts(clips):
    num_real = sum(1 for c in clips if c.get('label', 0) == 0)
    num_fake = sum(1 for c in clips if c.get('label', 1) == 1)
    return {'real': num_real, 'fake': num_fake}


def _count_unique_videos(clips):
    """统计clips中唯一的视频数量"""
    unique_videos = set()
    for clip in clips:
        video_id = clip.get('raw_rel_path', '')
        if video_id:
            unique_videos.add(video_id)
    return len(unique_videos)


class ClipTransform:
    """可被pickle的clip级变换：统一空间尺寸到 input_size，时间长度到 n_segment。"""
    def __init__(self, input_size: int = 224, n_segment: int = 8):
        self.input_size = int(input_size)
        self.n_segment = int(n_segment)

    def __call__(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        # clip_tensor: (C, T, H, W)
        C, T, H, W = clip_tensor.shape
        if C == 0 or T == 0:
            return torch.zeros((3, self.n_segment, self.input_size, self.input_size), dtype=torch.float32)

        # resize spatial to target size
        x = clip_tensor.unsqueeze(0)                 # (1, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)                 # (1, T, C, H, W)
        x = x.reshape(T, C, H, W)                    # (T, C, H, W)
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        x = x.reshape(1, T, C, self.input_size, self.input_size).permute(0, 2, 1, 3, 4).squeeze(0)  # (C, T, S, S)

        # adjust temporal length
        if T < self.n_segment:
            last = x[:, -1:, :, :].repeat(1, self.n_segment - T, 1, 1)
            x = torch.cat([x, last], dim=1)
        elif T > self.n_segment:
            x = x[:, :self.n_segment, :, :]
        return x.contiguous()


def build_dataloaders(config):
    data_root = config.get('data_root', 'data')
    index_path = config.get('index_path', os.path.join(data_root, 'dataset_index.pkl'))
    batch_size = config.get('batch_size', 8)
    num_workers = config.get('num_workers', 4)
    allow_skip = config.get('allow_skip', True)
    use_fast_io = config.get('use_fast_io', True)
    prefetch_factor = int(config.get('prefetch_factor', 2))
    persistent_workers_cfg = config.get('persistent_workers')
    dataset_name_filter = config.get('dataset_name')  # 如 'FFPP'

    clips = _load_index(index_path)
    if dataset_name_filter:
        # 仅保留指定数据集的 clips
        clips = [c for c in clips if c.get('dataset_name', '').lower() == str(dataset_name_filter).lower()]
    
    # 尝试加载已保存的划分，否则重新划分并保存
    split_cache_path = config.get('split_cache_path', None)
    if split_cache_path and os.path.exists(split_cache_path):
        print(f"[Dataset Split] 加载已保存的划分: {split_cache_path}")
        with open(split_cache_path, 'rb') as f:
            split_data = pickle.load(f)
        train_clips = split_data['train_clips']
        val_clips = split_data['val_clips']
        test_clips = split_data['test_clips']
    else:
        train_clips, val_clips, test_clips = _split_clips_three(
            clips,
            val_ratio=config.get('val_ratio', 0.1),
            test_ratio=config.get('test_ratio', 0.1),
            seed=config.get('seed', 42)
        )
        # 保存划分结果
        if split_cache_path:
            print(f"[Dataset Split] 保存划分结果: {split_cache_path}")
            os.makedirs(os.path.dirname(split_cache_path), exist_ok=True)
            split_data = {
                'train_clips': train_clips,
                'val_clips': val_clips,
                'test_clips': test_clips,
                'split_params': {
                    'val_ratio': config.get('val_ratio', 0.1),
                    'test_ratio': config.get('test_ratio', 0.1),
                    'seed': config.get('seed', 42),
                    'dataset_name': dataset_name_filter
                }
            }
            with open(split_cache_path, 'wb') as f:
                pickle.dump(split_data, f)

    clip_transform = ClipTransform(input_size=int(config.get('input_size', 224)), n_segment=int(config.get('n_segment', 8)))
    train_ds = DeepfakeDataset(train_clips, data_root=data_root, transform=clip_transform, allow_skip=allow_skip, use_fast_io=use_fast_io)
    val_ds = DeepfakeDataset(val_clips, data_root=data_root, transform=clip_transform, allow_skip=allow_skip, use_fast_io=use_fast_io)

    # 采样器：可选启用类别平衡
    sampler = None
    sampler_cfg = config.get('sampler', {})
    if isinstance(sampler_cfg, str):
        sampler_type = sampler_cfg
        sampler_cfg = {'type': sampler_type}
    sampler_type = sampler_cfg.get('type', 'none')
    if sampler_type == 'balanced':
        # 为训练集构造 WeightedRandomSampler，使每个类别采样概率相近
        counts = _compute_class_counts(train_clips)
        num_real = max(1, counts['real'])
        num_fake = max(1, counts['fake'])
        # 权重与样本数量成反比
        w_real = 1.0 / num_real
        w_fake = 1.0 / num_fake
        sample_weights = []
        for c in train_clips:
            sample_weights.append(w_fake if c.get('label', 1) == 1 else w_real)
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Windows 下为避免多进程 pickling 问题，默认强制单进程加载（可在配置中覆盖 force_single_worker=false）
    if os.name == 'nt' and config.get('force_single_worker', True):
        num_workers = 0
    persistent_workers = bool(persistent_workers_cfg) if persistent_workers_cfg is not None else (num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        DeepfakeDataset(test_clips, data_root=data_root, transform=clip_transform, allow_skip=allow_skip, use_fast_io=use_fast_io),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    # 打印数据集统计信息（包含视频级划分信息）
    print(f"[Dataset Split] Total clips: {len(clips)}")
    print(f"[Dataset Split] Train: {len(train_clips)} clips, {_count_unique_videos(train_clips)} videos")
    print(f"[Dataset Split] Val: {len(val_clips)} clips, {_count_unique_videos(val_clips)} videos")
    print(f"[Dataset Split] Test: {len(test_clips)} clips, {_count_unique_videos(test_clips)} videos")
    # 返回类别计数用于设定损失权重
    return train_loader, val_loader, test_loader, _compute_class_counts(train_clips)


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = float(gamma)
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        # 基于 BCEWithLogits + focal 调制
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
        p = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, p, 1.0 - p)
        loss = (1.0 - pt).pow(self.gamma) * bce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def build_model_and_optim(config, device, class_counts=None):
    model_name = config.get('model_name', 'mobilenetv4')
    n_segment = config.get('n_segment', 8)
    fold_div = config.get('fold_div', 8)
    pretrained = config.get('pretrained', True)

    # 确定分类头输出维度
    loss_type = config.get('loss', 'bce')  # 'focal_bce' / 'bce' / 'ce'
    if loss_type.lower() == 'ce':
        num_classes = int(config.get('num_classes', 2))
    else:
        num_classes = int(config.get('num_classes', 1))  # BCE 默认单通道

    model = create_mntsm_model(
        model_name=model_name,
        pretrained=pretrained,
        n_segment=n_segment,
        fold_div=fold_div,
        num_classes=num_classes,
    )
    model = model.to(device)

    # 损失与优化器
    if loss_type.lower() == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        # 自动正样本权重（pos=1=fake）。pos_weight = neg/pos
        pos_weight = None
        cb_cfg = config.get('class_balance', {})
        if isinstance(cb_cfg, bool):
            cb_cfg = {'auto_pos_weight': cb_cfg}
        if cb_cfg.get('auto_pos_weight', False) and class_counts is not None:
            num_pos = max(1, class_counts.get('fake', 1))
            num_neg = max(1, class_counts.get('real', 1))
            # 动态计算：sqrt(neg/pos)，更稳定，避免过度放大正样本梯度
            pw = math.sqrt(float(num_neg) / float(num_pos))
            pos_weight = torch.tensor([pw], device=device)
        if loss_type.lower() == 'focal_bce':
            gamma = float(cb_cfg.get('gamma', 2.0))
            criterion = FocalBCEWithLogitsLoss(gamma=gamma, pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 0.0)
    optimizer_name = str(config.get('optimizer', 'adamw')).lower()
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine + Warmup 调度（按 epoch 调整）
    sched_cfg = config.get('lr_scheduler', {}) or {}
    scheduler = None
    if str(sched_cfg.get('type', '')).lower() == 'cosine':
        total_epochs = int(config.get('epochs', 50))
        warmup_epochs = int(sched_cfg.get('warmup_epochs', 0))
        min_lr = float(sched_cfg.get('min_lr', 0.0))

        def lr_lambda(epoch_idx):
            # epoch_idx 从 0 开始
            if epoch_idx < warmup_epochs:
                return (epoch_idx + 1) / max(1, warmup_epochs)
            t = (epoch_idx - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            # 余弦从1->0，映射到 [min_lr/lr, 1]
            cos_scale = 0.5 * (1.0 + math.cos(math.pi * t))
            target = min_lr / max(1e-12, lr)
            return target + (1.0 - target) * cos_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return model, criterion, optimizer, scheduler


def save_checkpoint(state, ckpt_dir, filename):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(state, os.path.join(ckpt_dir, filename))


def train_from_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    set_seed(config.get('seed', 42))
    use_gpu = torch.cuda.is_available() and config.get('use_gpu', True)
    device = torch.device('cuda' if use_gpu else 'cpu')
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    exp_name = config.get('exp_name', 'MNTSM')
    log_dir = config.get('log_dir', 'experiments/logs')
    use_tb = config.get('tensorboard', False)
    use_wandb = config.get('wandb', {}).get('enable', False)
    wandb_project = config.get('wandb', {}).get('project')
    wandb_run_name = config.get('wandb', {}).get('run_name')

    # 若指定数据集名，则将日志与检查点写入对应子目录
    ds_name = config.get('dataset_name')
    if ds_name:
        log_dir = os.path.join(log_dir, ds_name)

    with experiment_logger(exp_name, log_dir=log_dir, use_tensorboard=use_tb,
                           use_wandb=use_wandb, wandb_project=wandb_project,
                           wandb_run_name=wandb_run_name, config=config,
                           use_console=bool(config.get('console_log', True)),
                           overwrite=bool(config.get('overwrite_log', True))) as logger:
        logger.info(f"Using device: {device}")

        # Dataloaders
        train_loader, val_loader, test_loader, class_counts = build_dataloaders(config)
        logger.info(f"Train clips: {len(train_loader.dataset)} | Val clips: {len(val_loader.dataset)} | Test clips: {len(test_loader.dataset)}")
        logger.info(f"Train class counts -> real: {class_counts.get('real', 0)}, fake: {class_counts.get('fake', 0)}")

        # Model/Optim
        model, criterion, optimizer, scheduler = build_model_and_optim(config, device, class_counts=class_counts)
        n_segment = config.get('n_segment', 8)

        # 训练循环
        epochs = config.get('epochs', 10)
        ckpt_dir = config.get('ckpt_dir', 'experiments/checkpoints')
        if ds_name:
            ckpt_dir = os.path.join(ckpt_dir, ds_name)
        best_auc = -1.0
        amp = bool(config.get('amp', True)) and device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

        # 早停参数
        es_cfg = config.get('early_stop', {}) or {}
        es_metric = es_cfg.get('metric', 'video_auc')
        es_mode = es_cfg.get('mode', 'max')
        es_patience = int(es_cfg.get('patience', 8))
        es_min_delta = float(es_cfg.get('min_delta', 1e-3))
        best_metric = -float('inf') if es_mode == 'max' else float('inf')
        epochs_no_improve = 0
        best_ckpt_state = None

        # 恢复训练（可选）
        start_epoch = 0
        resume_path = config.get('resume_from')
        if resume_path and os.path.isfile(resume_path):
            try:
                ckpt = torch.load(resume_path, map_location=device)
                model.load_state_dict(ckpt.get('model_state', {}))
                if 'optimizer_state' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state'])
                if 'scaler_state' in ckpt and scaler is not None:
                    scaler.load_state_dict(ckpt['scaler_state'])
                start_epoch = int(ckpt.get('epoch', 0))
                best_auc = float(ckpt.get('metrics', {}).get('video_auc', best_auc))
                # 恢复早停状态
                if 'best_metric' in ckpt:
                    best_metric = float(ckpt['best_metric'])
                if 'epochs_no_improve' in ckpt:
                    epochs_no_improve = int(ckpt['epochs_no_improve'])
                # 恢复调度器状态
                if scheduler is not None and 'scheduler_state' in ckpt:
                    try:
                        scheduler.load_state_dict(ckpt['scheduler_state'])
                    except Exception as e:
                        logger.warning(f"Failed to load scheduler state: {e}, will continue from epoch {start_epoch}")
                # 如果从best.pth恢复，设置best_ckpt_state以便最终测试使用
                # 这样可以确保即使没有在本次训练中更新best_auc，也能使用最佳模型进行测试
                if os.path.basename(resume_path) == 'best.pth':
                    best_ckpt_state = {
                        'epoch': start_epoch,
                        'model_state': ckpt.get('model_state', {}),
                        'optimizer_state': ckpt.get('optimizer_state'),
                        'scaler_state': ckpt.get('scaler_state'),
                        'config': ckpt.get('config'),
                        'metrics': ckpt.get('metrics', {})
                    }
                logger.info(f"Resumed from: {resume_path} (epoch={start_epoch}, best_auc={best_auc}, best_metric={best_metric}, epochs_no_improve={epochs_no_improve})")
            except Exception as e:
                logger.warning(f"Failed to resume from {resume_path}: {e}")

        grad_accum_steps = int(config.get('grad_accum_steps', 1))
        # 可选：冻结早期BN，缓解小batch/不平衡导致的统计漂移
        if bool(config.get('freeze_bn', False)):
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters(recurse=False):
                        p.requires_grad = False

        # 如果从断点恢复且调度器存在但未恢复状态，需要手动同步到正确epoch
        if scheduler is not None and start_epoch > 0:
            # 检查调度器是否已恢复状态（通过last_epoch判断）
            # 如果LambdaLR的last_epoch为-1，说明是新建的调度器，需要手动同步
            if hasattr(scheduler, 'last_epoch') and scheduler.last_epoch == -1:
                # 手动将调度器步进到恢复的epoch位置
                for _ in range(start_epoch):
                    try:
                        scheduler.step()
                    except Exception:
                        break

        for epoch in range(start_epoch + 1, epochs + 1):
            avg_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                n_segment=n_segment, aggregate=config.get('aggregate', 'mean'), amp=amp, scaler=scaler,
                grad_accum_steps=grad_accum_steps
            )
            metrics = evaluate(model, val_loader, device, n_segment=n_segment, aggregate=config.get('aggregate', 'mean'), amp=amp)

            logger.info(f"Epoch {epoch}/{epochs} | loss: {avg_loss:.4f}")
            logger.log_metrics({
                'train_loss': avg_loss,
                **metrics
            }, step=epoch)

            # 调度器按epoch步进
            if scheduler is not None:
                try:
                    scheduler.step()
                except Exception:
                    pass

            # 保存最新与最佳（包含早停状态和调度器状态）
            checkpoint_data = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict() if scaler is not None else None,
                'config': config,
                'metrics': metrics,
                'best_metric': best_metric,
                'epochs_no_improve': epochs_no_improve
            }
            if scheduler is not None:
                checkpoint_data['scheduler_state'] = scheduler.state_dict()
            save_checkpoint(checkpoint_data, ckpt_dir, 'last.pth')

            if metrics.get('video_auc', -1) > best_auc:
                best_auc = metrics['video_auc']
                best_ckpt_state = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scaler_state': scaler.state_dict() if scaler is not None else None,
                    'config': config,
                    'metrics': metrics,
                    'best_metric': best_metric,
                    'epochs_no_improve': epochs_no_improve
                }
                if scheduler is not None:
                    best_ckpt_state['scheduler_state'] = scheduler.state_dict()
                save_checkpoint(best_ckpt_state, ckpt_dir, 'best.pth')

            # 早停逻辑
            cur = metrics.get(es_metric)
            if cur is not None and es_patience > 0:
                improved = (cur > best_metric + es_min_delta) if es_mode == 'max' else (cur < best_metric - es_min_delta)
                if improved:
                    best_metric = cur
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= es_patience:
                        logger.info(f"Early stopping at epoch {epoch}: no improvement in {es_patience} epochs on {es_metric}.")
                        break

        # 使用最佳权重（如果有）进行最终测试评估
        if best_ckpt_state is not None:
            try:
                model.load_state_dict(best_ckpt_state['model_state'])
            except Exception:
                pass
        # 最终在测试集评估
        test_metrics = evaluate(model, test_loader, device, n_segment=n_segment, aggregate=config.get('aggregate', 'mean'), amp=amp)
        logger.log_metrics({**{f"test_{k}": v for k, v in test_metrics.items()}}, step=epochs)
        logger.info(f"Training finished. Best video AUC: {best_auc:.4f} | Test video AUC: {test_metrics.get('video_auc', 0.0):.4f}")
