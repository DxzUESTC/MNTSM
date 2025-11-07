"""泛化性测试模块：在另一个数据集上测试已训练的模型"""
import os
import pickle
import yaml
import torch
import argparse

from .dataset_loader import DeepfakeDataset
from .evaluator import evaluate
from ..models.mobilenetv4_tsm import create_mntsm_model
from ..utils.logger import experiment_logger
from .trainer import ClipTransform


def load_model_from_checkpoint(checkpoint_path, config, device):
    """从检查点加载模型"""
    print(f"[INFO] 加载模型检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从检查点或配置中获取模型参数
    model_config = checkpoint.get('config', config)
    
    # 如果检查点中有配置，优先使用检查点的配置（尤其是模型结构相关参数）
    model_name = model_config.get('model_name', config.get('model_name', 'mobilenetv4'))
    n_segment = model_config.get('n_segment', config.get('n_segment', 8))
    fold_div = model_config.get('fold_div', config.get('fold_div', 8))
    pretrained = model_config.get('pretrained', config.get('pretrained', True))
    cache_dir = config.get('model_cache_dir', None)
    
    # 确定分类头输出维度
    loss_type = model_config.get('loss', config.get('loss', 'bce'))
    if loss_type.lower() == 'ce':
        num_classes = int(model_config.get('num_classes', config.get('num_classes', 2)))
    else:
        num_classes = int(model_config.get('num_classes', config.get('num_classes', 1)))
    
    # 创建模型
    model = create_mntsm_model(
        model_name=model_name,
        pretrained=False,  # 不使用预训练，因为我们要加载检查点
        n_segment=n_segment,
        fold_div=fold_div,
        num_classes=num_classes,
        cache_dir=cache_dir,
    )
    
    # 加载权重
    model_state = checkpoint.get('model_state', checkpoint)
    try:
        model.load_state_dict(model_state)
        print("[INFO] 成功加载模型权重")
    except Exception as e:
        print(f"[WARNING] 直接加载失败，尝试匹配键名: {e}")
        # 尝试处理键名不匹配的情况
        if isinstance(model_state, dict):
            # 移除 'module.' 前缀（如果模型是用 DataParallel 保存的）
            new_state_dict = {}
            for k, v in model_state.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
            print("[INFO] 使用键名匹配加载模型权重")
    
    model = model.to(device)
    model.eval()
    
    return model, {
        'n_segment': n_segment,
        'aggregate': model_config.get('aggregate', config.get('aggregate', 'mean')),
        'input_size': model_config.get('input_size', config.get('input_size', 224)),
    }


def build_test_dataloader(config, split_path, data_root='data'):
    """构建测试数据加载器"""
    print(f"[INFO] 加载数据集划分: {split_path}")
    with open(split_path, 'rb') as f:
        split_data = pickle.load(f)
    
    # 使用测试集
    test_clips = split_data.get('test_clips', [])
    if len(test_clips) == 0:
        print("[WARNING] 测试集为空，尝试使用验证集...")
        test_clips = split_data.get('val_clips', [])
    
    if len(test_clips) == 0:
        raise ValueError("数据集划分文件中没有找到测试集或验证集")
    
    print(f"[INFO] 测试集包含 {len(test_clips)} 个 clips")
    
    # 创建变换
    input_size = config.get('input_size', 224)
    n_segment = config.get('n_segment', 8)
    clip_transform = ClipTransform(input_size=input_size, n_segment=n_segment)
    
    # 创建数据集和数据加载器
    dataset = DeepfakeDataset(
        test_clips,
        data_root=data_root,
        transform=clip_transform,
        allow_skip=config.get('allow_skip', True),
        use_fast_io=config.get('use_fast_io', True)
    )
    
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 0)
    if os.name == 'nt' and config.get('force_single_worker', True):
        num_workers = 0
    
    persistent_workers = num_workers > 0
    prefetch_factor = int(config.get('prefetch_factor', 2)) if num_workers > 0 else None
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    
    return dataloader


def generalization_test(config_path: str):
    """执行泛化性测试"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    use_gpu = torch.cuda.is_available() and config.get('use_gpu', True)
    device = torch.device('cuda' if use_gpu else 'cpu')
    if use_gpu:
        torch.backends.cudnn.benchmark = True
    
    # 设置输出目录
    exp_name = config.get('exp_name', 'Generalization_Test')
    log_dir = config.get('log_dir', 'generalization/experiments/logs')
    ckpt_dir = config.get('ckpt_dir', 'generalization/experiments/checkpoints')
    
    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 设置日志
    use_tb = config.get('tensorboard', False)
    use_wandb = config.get('wandb', {}).get('enable', False)
    wandb_project = config.get('wandb', {}).get('project')
    wandb_run_name = config.get('wandb', {}).get('run_name')
    
    with experiment_logger(
        exp_name,
        log_dir=log_dir,
        use_tensorboard=use_tb,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        config=config,
        use_console=True,
        overwrite=True
    ) as logger:
        logger.info(f"使用设备: {device}")
        logger.info(f"泛化性测试配置: {config_path}")
        
        # 加载模型
        checkpoint_path = config.get('checkpoint_path')
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise ValueError(f"检查点文件不存在: {checkpoint_path}")
        
        model, model_params = load_model_from_checkpoint(checkpoint_path, config, device)
        logger.info(f"模型参数: n_segment={model_params['n_segment']}, "
                   f"aggregate={model_params['aggregate']}, "
                   f"input_size={model_params['input_size']}")
        
        # 构建测试数据加载器
        split_path = config.get('test_split_path')
        if not split_path or not os.path.exists(split_path):
            raise ValueError(f"数据集划分文件不存在: {split_path}")
        
        data_root = config.get('data_root', 'data')
        test_loader = build_test_dataloader(config, split_path, data_root=data_root)
        logger.info(f"测试集大小: {len(test_loader.dataset)} clips")
        
        # 执行评估
        logger.info("开始评估...")
        amp = bool(config.get('amp', False)) and device.type == 'cuda'
        metrics = evaluate(
            model,
            test_loader,
            device,
            n_segment=model_params['n_segment'],
            aggregate=model_params['aggregate'],
            amp=amp
        )
        
        # 记录结果
        logger.info("=" * 60)
        logger.info("泛化性测试结果:")
        logger.info("=" * 60)
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")
        logger.info("=" * 60)
        
        # 记录指标
        logger.log_metrics(metrics, step=0)
        
        # 保存结果
        result_path = os.path.join(ckpt_dir, 'generalization_results.txt')
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write("泛化性测试结果\n")
            f.write("=" * 60 + "\n")
            f.write(f"模型检查点: {checkpoint_path}\n")
            f.write(f"测试数据集: {config.get('test_split_path')}\n")
            f.write(f"测试集大小: {len(test_loader.dataset)} clips\n")
            f.write("=" * 60 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write("=" * 60 + "\n")
        
        logger.info(f"结果已保存到: {result_path}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='泛化性测试：在另一个数据集上测试已训练的模型')
    parser.add_argument('--config', type=str, required=True,
                       help='测试配置文件路径')
    args = parser.parse_args()
    
    generalization_test(args.config)


if __name__ == '__main__':
    main()

