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
