"""日志工具模块

提供统一日志接口，并可选支持 TensorBoard 与 Weights & Biases。
"""
import logging
import os
from contextlib import contextmanager


class ExperimentLogger:
    """统一实验日志封装。

    Args:
        name (str): 实验名称（同时作为log文件名前缀）。
        log_dir (str): 文本日志目录。
        use_tensorboard (bool): 是否启用 TensorBoard。
        use_wandb (bool): 是否启用 Weights & Biases。
        wandb_project (str): W&B 项目名。
        wandb_run_name (str): W&B run 名称。
        config (dict): 可选，记录的配置超参数。
    """

    def __init__(self, name,
                 log_dir='experiments/logs',
                 use_tensorboard=False,
                 use_wandb=False,
                 wandb_project=None,
                 wandb_run_name=None,
                 config=None,
                 use_console=True,
                 auto_system_info=True,
                 overwrite=False):
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            log_path = os.path.join(log_dir, f"{name}.log")
            mode = 'w' if overwrite else 'a'
            fh = logging.FileHandler(log_path, mode=mode, encoding='utf-8')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            if use_console:
                ch = logging.StreamHandler()
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)

        # TensorBoard
        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, 'tb')
                os.makedirs(tb_dir, exist_ok=True)
                self.tb = SummaryWriter(log_dir=tb_dir)
                if config is not None:
                    self.tb.add_text('config', str(config))
            except Exception as e:
                self.logger.warning(f"TensorBoard 初始化失败: {e}")
                self.tb = None

        # Weights & Biases
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                init_kwargs = {}
                if wandb_project:
                    init_kwargs['project'] = wandb_project
                if wandb_run_name:
                    init_kwargs['name'] = wandb_run_name
                if config is not None:
                    init_kwargs['config'] = config
                self.wandb.init(**init_kwargs)
            except Exception as e:
                self.logger.warning(f"W&B 初始化失败: {e}")
                self.wandb = None

        # 自动记录系统/时间/硬件信息
        if auto_system_info:
            try:
                self.log_system_info()
            except Exception:
                pass

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def log_metrics(self, metrics: dict, step: int = None, prefix: str = None):
        data = metrics if prefix is None else {f"{prefix}/{k}": v for k, v in metrics.items()}
        # 文本日志
        self.info(" | ".join([f"{k}: {v:.6f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in data.items()]))
        # TensorBoard
        if self.tb is not None:
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, global_step=step)
        # W&B
        if self.wandb is not None:
            try:
                self.wandb.log(data if step is None else {**data, 'step': step})
            except Exception:
                pass

    # 额外便捷日志方法
    def log_config_summary(self, config: dict):
        try:
            self.info("配置摘要:")
            for k, v in (config or {}).items():
                self.info(f"  {k}: {v}")
            if self.tb is not None and config is not None:
                self.tb.add_text('config_summary', str(config))
        except Exception:
            pass

    def log_hyperparameters(self, hyperparams: dict, section_name: str = "训练超参数"):
        """记录训练的超参数和关键参数
        
        Args:
            hyperparams: 超参数字典，键为参数名，值为参数值
            section_name: 章节名称，用于标识不同的超参数组
        """
        try:
            self.info("=" * 80)
            self.info(f"{section_name}:")
            self.info("-" * 80)
            
            # 按类别分组显示（如果有嵌套字典）
            if isinstance(hyperparams, dict):
                for key, value in hyperparams.items():
                    if isinstance(value, dict):
                        # 嵌套字典，显示子标题
                        self.info(f"\n  [{key}]:")
                        for sub_key, sub_value in value.items():
                            formatted_value = self._format_hyperparameter_value(sub_value)
                            self.info(f"    {sub_key}: {formatted_value}")
                    else:
                        formatted_value = self._format_hyperparameter_value(value)
                        self.info(f"  {key}: {formatted_value}")
            
            self.info("=" * 80)
            
            # 同时记录到TensorBoard和W&B
            if self.tb is not None:
                self.tb.add_text(section_name, self._dict_to_markdown(hyperparams))
            
            if self.wandb is not None:
                # W&B会自动记录config，这里额外记录为超参数
                flat_params = self._flatten_dict(hyperparams)
                self.wandb.config.update(flat_params)
                
        except Exception as e:
            self.warning(f"记录超参数失败: {e}")

    def _format_hyperparameter_value(self, value):
        """格式化超参数值以便显示"""
        if isinstance(value, (int, float)):
            if isinstance(value, float) and abs(value) < 0.001:
                return f"{value:.6e}"
            elif isinstance(value, float):
                return f"{value:.6f}"
            else:
                return str(value)
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (list, tuple)):
            if len(value) > 10:
                return f"{value[:5]} ... ({len(value)} items)"
            return str(value)
        elif value is None:
            return "None"
        else:
            return str(value)

    def _dict_to_markdown(self, d: dict, indent: int = 0) -> str:
        """将字典转换为Markdown格式字符串"""
        lines = []
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}**{key}**:")
                lines.append(self._dict_to_markdown(value, indent + 1))
            else:
                formatted_value = self._format_hyperparameter_value(value)
                lines.append(f"{prefix}- **{key}**: {formatted_value}")
        return "\n".join(lines)

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """展平嵌套字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def log_model_info(self, model, optimizer=None, scheduler=None):
        """记录模型、优化器和调度器的详细信息"""
        try:
            self.info("=" * 80)
            self.info("模型信息:")
            self.info("-" * 80)
            
            # 模型参数统计
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.info(f"  总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
            self.info(f"  可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
            self.info(f"  冻结参数量: {(total_params - trainable_params):,}")
            
            # 模型名称/类型
            model_name = model.__class__.__name__
            self.info(f"  模型类型: {model_name}")
            
            # 优化器信息
            if optimizer is not None:
                self.info(f"\n优化器信息:")
                opt_name = optimizer.__class__.__name__
                self.info(f"  类型: {opt_name}")
                if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                    lr = optimizer.param_groups[0].get('lr', 'N/A')
                    weight_decay = optimizer.param_groups[0].get('weight_decay', 'N/A')
                    self.info(f"  学习率: {lr}")
                    self.info(f"  权重衰减: {weight_decay}")
            
            # 调度器信息
            if scheduler is not None:
                self.info(f"\n学习率调度器:")
                sched_name = scheduler.__class__.__name__
                self.info(f"  类型: {sched_name}")
                if hasattr(scheduler, 'last_epoch'):
                    self.info(f"  当前epoch: {scheduler.last_epoch}")
            
            self.info("=" * 80)
            
            # 记录到TensorBoard
            if self.tb is not None:
                self.tb.add_scalar('model/total_params', total_params, 0)
                self.tb.add_scalar('model/trainable_params', trainable_params, 0)
            
        except Exception as e:
            self.warning(f"记录模型信息失败: {e}")

    def log_dataset_summary(self, train_count: int, val_count: int, class_counts: dict = None):
        try:
            self.info("数据集摘要:")
            self.info(f"  训练clips: {train_count}")
            self.info(f"  验证clips: {val_count}")
            if class_counts is not None:
                real = class_counts.get('real', 0)
                fake = class_counts.get('fake', 0)
                self.info(f"  训练类别计数 -> real: {real}, fake: {fake}")
                if self.tb is not None:
                    self.tb.add_scalar('data/train_real', real, 0)
                    self.tb.add_scalar('data/train_fake', fake, 0)
        except Exception:
            pass

    def log_system_info(self):
        import datetime, platform
        self.info("系统/硬件信息:")
        self.info(f"  时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"  平台: {platform.platform()}")
        try:
            import torch
            self.info(f"  CUDA 可用: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                self.info(f"  GPU: {torch.cuda.get_device_name(0)}")
                self.info(f"  CUDA 版本: {getattr(torch.version, 'cuda', 'unknown')}")
                self.info(f"  cuDNN 版本: {getattr(torch.backends.cudnn, 'version', lambda: 'unknown')()}")
        except Exception:
            pass
        try:
            import psutil
            mem = psutil.virtual_memory()
            self.info(f"  内存: {round(mem.total/1024**3, 2)} GB 总, {round(mem.available/1024**3, 2)} GB 可用")
            self.info(f"  CPU: 逻辑核 {psutil.cpu_count(logical=True)} / 物理核 {psutil.cpu_count(logical=False)}")
        except Exception:
            pass

    def close(self):
        if self.tb is not None:
            try:
                self.tb.flush()
                self.tb.close()
            except Exception:
                pass
        if self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass


def get_logger(name, log_dir='experiments/logs'):
    """兼容旧接口：仅返回标准 logging.Logger。"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


@contextmanager
def experiment_logger(name, **kwargs):
    """上下文管理器：创建 ExperimentLogger 并自动关闭资源。"""
    el = ExperimentLogger(name, **kwargs)
    try:
        yield el
    finally:
        el.close()
