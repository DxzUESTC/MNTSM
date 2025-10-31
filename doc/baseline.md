# MobileNet V4 模型的对比

timm库中的预训练模型：

| model_name                                        | params_M | input_size        | macs_G   | note |
| ------------------------------------------------- | -------- | ----------------- | -------- | ---- |
| mobilenetv4_conv_aa_large.e230_r384_in12k         | 46.45    | (3, 384, 384)     | 7.02     |      |
| mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k | 32.59    | (3, 384, 384)     | 7.01     |      |
| mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k | 32.59    | (3, 448, 448)     | 9.54     |      |
| mobilenetv4_conv_aa_large.e600_r384_in1k          | 32.59    | (3, 384, 384)     | 7.01     |      |
| mobilenetv4_conv_blur_medium.e500_r224_in1k       | 9.72     | (3, 224, 224)     | 1.20     |      |
| mobilenetv4_conv_large.e500_r256_in1k             | 32.59    | (3, 256, 256)     | 2.84     |      |
| mobilenetv4_conv_large.e600_r384_in1k             | 32.59    | (3, 384, 384)     | 6.38     |      |
| mobilenetv4_conv_medium.e180_ad_r384_in12k        | 23.58    | (3, 384, 384)     | 2.44     |      |
| mobilenetv4_conv_medium.e180_r384_in12k           | 23.58    | (3, 384, 384)     | 2.44     |      |
| mobilenetv4_conv_medium.e250_r384_in12k           | 23.58    | (3, 384, 384)     | 2.44     |      |
| mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k   | 9.72     | (3, 384, 384)     | 2.43     |      |
| mobilenetv4_conv_medium.e500_r224_in1k            | 9.72     | (3, 224, 224)     | 0.83     |      |
| mobilenetv4_conv_medium.e500_r256_in1k            | 9.72     | (3, 256, 256)     | 1.08     |      |
| mobilenetv4_conv_small.e1200_r224_in1k            | 3.77     | (3, 224, 224)     | 0.19     |      |
| mobilenetv4_conv_small.e2400_r224_in1k            | 3.77     | (3, 224, 224)     | 0.19     |      |
| mobilenetv4_conv_small.e3600_r256_in1k            | 3.77     | (3, 256, 256)     | 0.24     |      |
| **mobilenetv4_conv_small_050.e3000_r224_in1k**    | **2.24** | **(3, 224, 224)** | **0.06** | √    |
| mobilenetv4_hybrid_large.e600_r384_in1k           | 37.76    | (3, 384, 384)     | 7.37     |      |
| mobilenetv4_hybrid_large.ix_e600_r384_in1k        | 37.76    | (3, 384, 384)     | 7.37     |      |
| mobilenetv4_hybrid_medium.e200_r256_in12k         | 24.94    | (3, 256, 256)     | 1.24     |      |
| mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k | 11.07    | (3, 256, 256)     | 1.23     |      |
| mobilenetv4_hybrid_medium.e500_r224_in1k          | 11.07    | (3, 224, 224)     | 0.94     |      |
| mobilenetv4_hybrid_medium.ix_e550_r256_in1k       | 11.07    | (3, 256, 256)     | 1.23     |      |
| mobilenetv4_hybrid_medium.ix_e550_r384_in1k       | 11.07    | (3, 384, 384)     | 2.76     |      |

模型名各个字段的解析：

| 段                                    | 含义                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| `mobilenetv4`                         | 模型系列                                                     |
| `conv` / `hybrid`                     | 主干结构`conv` = 纯卷积网络，`hybrid` = 卷积+Transformer混合 |
| `small` / `medium` / `large` / `_050` | 模型宽度/参数量大小small < medium < large                    |
| `aa` / `blur` / `ad`                  | 数据增强策略`aa` = AutoAugment, `blur` = 模糊训练, `ad` = adv data? |
| `r224` / `r256` / `r384` / `r448`     | 输入分辨率（H×W）                                            |
| `in1k` / `in12k`                      | ImageNet 数据集规模1k = ImageNet-1K, 12k = ImageNet-12K      |
| `_ft_in1k`                            | fine-tuned on ImageNet-1K通常比原始权重稍好                  |
| `e600` / `e2400`                      | 训练 epochs 或模型版本号                                     |



# 快速验证

使用的预训练模型：**mobilenetv4_conv_small_050.e3000_r224_in1k**

数据集：FF++中c23压缩版本，已裁剪出人脸，人脸尺寸 224x224，文件地址 D:\Share\mobilenet_tsm\dataset\faces_224\xxx\c23



## MobileNet v4 快速验证

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import glob
import logging
from datetime import datetime
import random

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    """设置所有随机数生成器的种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # 确保CUDNN的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在程序开始时设置种子
set_seed(42)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数配置
class Config:
    # 随机种子
    seed = 42
    
    # 数据集配置
    data_root = r"d:\Share\mobilenet_tsm\dataset\faces_224"
    
    # 训练配置
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    num_workers = 4
    
    # CUDA优化配置
    use_amp = False  # 使用混合精度训练
    pin_memory = True  # DataLoader使用pin_memory加速
    
    # 早停配置
    patience = 10  # 早停耐心值
    
    # 模型配置
    model_name = 'mobilenetv4_conv_small_050.e3000_r224_in1k'
    num_classes = 2  # 真实 vs 伪造
    
    # 数据划分比例
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    # 日志配置
    log_dir = r"d:\Share\mobilenet_tsm\output\baseline\mobilenet_small\logs\251017"
    
    # 模型保存配置
    checkpoint_dir = r"d:\Share\mobilenet_tsm\output\baseline\mobilenet_small\checkpoints\251017"
    save_best_only = True  # 是否只保存最佳模型
    save_every_n_epochs = 5  # 每N个epoch保存一次(0表示不启用)

config = Config()


def setup_logger(log_dir):
    """
    设置日志系统
    创建日志目录和日志文件，配置日志格式
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件创建: {log_file}")
    
    return logger


def save_checkpoint(model, optimizer, epoch, best_auc, checkpoint_dir, filename, logger=None):
    """
    保存训练检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        best_auc: 最佳AUC分数
        checkpoint_dir: 检查点保存目录
        filename: 检查点文件名
        logger: 日志记录器
    """
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 准备检查点数据
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
        'model_name': config.model_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存路径
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    
    msg = f"检查点已保存: {checkpoint_path}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu', logger=None):
    """
    加载训练检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 模型
        optimizer: 优化器(可选)
        device: 设备
        logger: 日志记录器
    
    Returns:
        epoch, best_auc: 恢复的epoch和最佳AUC
    """
    if not os.path.exists(checkpoint_path):
        msg = f"检查点文件不存在: {checkpoint_path}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return 0, 0.0
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_auc = checkpoint.get('best_auc', 0.0)
    
    msg = f"检查点已加载: {checkpoint_path} (Epoch: {epoch}, Best AUC: {best_auc:.4f})"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    return epoch, best_auc


# 自定义数据集类
class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: 图片路径列表
            labels: 标签列表 (0=真实, 1=伪造)
            transform: 数据增强
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 读取图片
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_dataset(data_root, logger=None):
    """
    加载数据集并划分训练/验证/测试集
    文件名格式: XYYY_ZZZ_frame_NNNN_face.jpg
    - X (第1位): 伪造类型 (0=真实, 1-5=不同伪造方法)
    - YYY (后3位): 视频ID
    按视频ID划分: 前80%训练, 中10%验证, 后10%测试
    """
    # 收集所有图片并提取视频ID
    video_dict = {}  # {video_id: [(image_path, label), ...]}
    
    # 遍历所有类别文件夹
    categories = ['original', 'deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTexture']
    
    for category in categories:
        category_path = os.path.join(data_root, category, 'c23')
        if not os.path.exists(category_path):
            msg = f"警告: 路径不存在 {category_path}"
            print(msg)
            if logger:
                logger.warning(msg)
            continue
        
        # 获取所有图片
        images = glob.glob(os.path.join(category_path, '*.jpg'))
        
        for img_path in images:
            filename = os.path.basename(img_path)
            # 提取前4位数字: XYYY
            parts = filename.split('_')
            if len(parts) >= 2 and parts[0].isdigit():
                full_id = parts[0]  # XYYY
                if len(full_id) == 4:
                    # 第1位是伪造类型标识
                    fake_type = int(full_id[0])
                    # 后3位是视频ID
                    video_id = full_id[1:]
                    
                    # 标签: 0开头的是真实(label=0), 其他是伪造(label=1)
                    label = 0 if fake_type == 0 else 1
                    
                    if video_id not in video_dict:
                        video_dict[video_id] = []
                    video_dict[video_id].append((img_path, label))
        
        msg = f"{category}: {len(images)} 张图片"
        print(msg)
        if logger:
            logger.info(msg)
    
    # 按视频ID排序
    video_ids = sorted(video_dict.keys())
    msg = f"\n总共 {len(video_ids)} 个视频ID"
    print(msg)
    if logger:
        logger.info(msg)
    
    # 计算划分点
    total_videos = len(video_ids)
    train_end = int(total_videos * config.train_ratio)
    val_end = int(total_videos * (config.train_ratio + config.val_ratio))
    
    # 划分视频ID
    train_video_ids = video_ids[:train_end]
    val_video_ids = video_ids[train_end:val_end]
    test_video_ids = video_ids[val_end:]
    
    msg = f"训练视频ID数: {len(train_video_ids)}"
    print(msg)
    if logger:
        logger.info(msg)
    
    msg = f"验证视频ID数: {len(val_video_ids)}"
    print(msg)
    if logger:
        logger.info(msg)
    
    msg = f"测试视频ID数: {len(test_video_ids)}"
    print(msg)
    if logger:
        logger.info(msg)
    
    # 根据视频ID收集图片
    def collect_images(video_id_list):
        images = []
        labels = []
        for vid in video_id_list:
            for img_path, label in video_dict[vid]:
                images.append(img_path)
                labels.append(label)
        return images, labels
    
    train_images, train_labels = collect_images(train_video_ids)
    val_images, val_labels = collect_images(val_video_ids)
    test_images, test_labels = collect_images(test_video_ids)
    
    msg = f"\n数据集划分:"
    print(msg)
    if logger:
        logger.info(msg)
    
    msg = f"训练集: {len(train_images)} 张 (真实: {train_labels.count(0)}, 伪造: {train_labels.count(1)})"
    print(msg)
    if logger:
        logger.info(msg)
    
    msg = f"验证集: {len(val_images)} 张 (真实: {val_labels.count(0)}, 伪造: {val_labels.count(1)})"
    print(msg)
    if logger:
        logger.info(msg)
    
    msg = f"测试集: {len(test_images)} 张 (真实: {test_labels.count(0)}, 伪造: {test_labels.count(1)})"
    print(msg)
    if logger:
        logger.info(msg)
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def get_transforms(is_training=True):
    """获取数据增强"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_model(num_classes=2):
    """创建 MobileNetV4 模型"""
    print(f"加载模型: {config.model_name}")
    model = timm.create_model(config.model_name, pretrained=True, num_classes=num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    use_amp = scaler is not None
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 统计
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1]  # 获取伪造类的概率
        preds = torch.argmax(outputs, dim=1)
        
        all_probs.extend(probs.cpu().detach().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    # 计算指标
    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    
    return avg_loss, acc, auc, f1


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    
    return avg_loss, acc, auc, f1


def main():
    # 0. 设置日志系统
    logger = setup_logger(config.log_dir)
    
    logger.info("=" * 60)
    logger.info("开始训练 - MobileNetV4 Deepfake检测")
    logger.info("=" * 60)
    
    # 记录随机种子
    logger.info(f"\n随机种子: {config.seed}")
    
    # 1. 记录设备信息
    logger.info("\n设备信息:")
    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # 启用CUDNN加速
        # 注意：由于设置了随机种子，CUDNN的benchmark被禁用以确保可重复性
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    else:
        logger.info("CUDA不可用，将使用CPU训练")
    
    # 2. 记录超参数设置
    logger.info("\n" + "=" * 60)
    logger.info("超参数设置:")
    logger.info(f"随机种子: {config.seed}")
    logger.info(f"模型名称: {config.model_name}")
    logger.info(f"批次大小: {config.batch_size}")
    logger.info(f"最大训练轮数: {config.num_epochs}")
    logger.info(f"学习率: {config.learning_rate}")
    logger.info(f"工作线程数: {config.num_workers}")
    logger.info(f"混合精度训练: {config.use_amp}")
    logger.info(f"Pin Memory: {config.pin_memory}")
    logger.info(f"早停耐心值: {config.patience}")
    logger.info(f"数据划分比例 - 训练/验证/测试: {config.train_ratio}/{config.val_ratio}/{config.test_ratio}")
    logger.info("=" * 60)
    
    # 3. 加载数据集
    logger.info("\n加载数据集...")
    train_data, val_data, test_data = load_dataset(config.data_root, logger)
    
    # 4. 创建数据加载器
    train_dataset = DeepfakeDataset(train_data[0], train_data[1], 
                                    transform=get_transforms(is_training=True))
    val_dataset = DeepfakeDataset(val_data[0], val_data[1], 
                                  transform=get_transforms(is_training=False))
    test_dataset = DeepfakeDataset(test_data[0], test_data[1], 
                                   transform=get_transforms(is_training=False))
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=config.num_workers,
                             pin_memory=config.pin_memory and torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=config.num_workers,
                           pin_memory=config.pin_memory and torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=config.num_workers,
                            pin_memory=config.pin_memory and torch.cuda.is_available())
    
    # 5. 创建模型
    logger.info("\n" + "=" * 60)
    logger.info("创建模型...")
    logger.info(f"模型架构: {config.model_name}")
    model = create_model(num_classes=config.num_classes).to(device)
    
    # 记录模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")
    logger.info("=" * 60)
    
    # 6. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    logger.info(f"损失函数: CrossEntropyLoss")
    logger.info(f"优化器: Adam (lr={config.learning_rate})")
    
    # 7. 混合精度训练scaler
    scaler = None
    if config.use_amp and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        logger.info("启用混合精度训练 (AMP)")
    
    # 8. 训练循环
    logger.info("\n" + "=" * 60)
    logger.info("开始训练...")
    logger.info(f"检查点保存目录: {config.checkpoint_dir}")
    logger.info("=" * 60)
    best_auc = 0.0
    best_model_state = None
    best_checkpoint_path = None
    patience_counter = 0
    
    # 生成时间戳用于文件命名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch [{epoch+1}/{config.num_epochs}]")
        
        # 训练
        train_loss, train_acc, train_auc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler)
        
        # 验证
        val_loss, val_acc, val_auc, val_f1 = evaluate(
            model, val_loader, criterion, device)
        
        # 记录每轮的训练和验证指标
        logger.info(f"Train - Loss: {train_loss:.4f}, ACC: {train_acc:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, ACC: {val_acc:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
        
        # 早停逻辑 (基于AUC)
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info(f"✓ 新的最佳验证AUC: {best_auc:.4f}")
            
            # 保存最佳模型
            if config.save_best_only:
                best_filename = f"best_model_{timestamp}.pth"
                best_checkpoint_path = save_checkpoint(
                    model, optimizer, epoch+1, best_auc, 
                    config.checkpoint_dir, best_filename, logger
                )
        else:
            patience_counter += 1
            logger.info(f"未改善 ({patience_counter}/{config.patience})")
            
            if patience_counter >= config.patience:
                logger.info(f"\n早停触发! 在第 {epoch+1} 轮停止训练")
                logger.info(f"验证AUC在 {config.patience} 个epoch内未改善")
                break
        
        # 定期保存检查点
        if config.save_every_n_epochs > 0 and (epoch + 1) % config.save_every_n_epochs == 0:
            checkpoint_filename = f"checkpoint_epoch_{epoch+1}_{timestamp}.pth"
            save_checkpoint(
                model, optimizer, epoch+1, best_auc,
                config.checkpoint_dir, checkpoint_filename, logger
            )
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 9. 加载最佳模型并在测试集上评估
    logger.info("\n" + "=" * 60)
    logger.info("在测试集上评估最佳模型...")
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_auc, test_f1 = evaluate(
        model, test_loader, criterion, device)
    
    # 记录最终测试集结果
    logger.info("\n" + "=" * 60)
    logger.info("最终测试集结果:")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test ACC:  {test_acc:.4f}")
    logger.info(f"Test AUC:  {test_auc:.4f}")
    logger.info(f"Test F1:   {test_f1:.4f}")
    logger.info("=" * 60)
    
    # 保存最终模型(包含测试结果)
    final_filename = f"final_model_{timestamp}_testauc_{test_auc:.4f}.pth"
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_auc': best_auc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'model_name': config.model_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs,
            'train_ratio': config.train_ratio,
            'val_ratio': config.val_ratio,
            'test_ratio': config.test_ratio
        }
    }
    final_path = os.path.join(config.checkpoint_dir, final_filename)
    torch.save(final_checkpoint, final_path)
    logger.info(f"\n最终模型已保存: {final_path}")
    
    logger.info("\n训练完成!")
    logger.info(f"最佳验证AUC: {best_auc:.4f}")
    if best_checkpoint_path:
        logger.info(f"最佳模型路径: {best_checkpoint_path}")
    logger.info(f"最终模型路径: {final_path}")
    logger.info("=" * 60)


def load_model_for_inference(checkpoint_path, device='cuda'):
    """
    从检查点加载模型用于推理
    
    Args:
        checkpoint_path: 检查点文件路径
        device: 运行设备
    
    Returns:
        model: 加载好的模型
        checkpoint_info: 检查点信息字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型
    model_name = checkpoint.get('model_name', config.model_name)
    num_classes = config.num_classes
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 提取检查点信息
    checkpoint_info = {
        'model_name': model_name,
        'timestamp': checkpoint.get('timestamp', 'Unknown'),
        'best_val_auc': checkpoint.get('best_val_auc', checkpoint.get('best_auc', 0.0)),
        'test_auc': checkpoint.get('test_auc', None),
        'test_acc': checkpoint.get('test_acc', None),
        'test_f1': checkpoint.get('test_f1', None),
    }
    
    print(f"模型已加载: {checkpoint_path}")
    print(f"模型名称: {checkpoint_info['model_name']}")
    print(f"保存时间: {checkpoint_info['timestamp']}")
    if checkpoint_info['test_auc'] is not None:
        print(f"测试AUC: {checkpoint_info['test_auc']:.4f}")
        print(f"测试ACC: {checkpoint_info['test_acc']:.4f}")
        print(f"测试F1: {checkpoint_info['test_f1']:.4f}")
    else:
        print(f"最佳验证AUC: {checkpoint_info['best_val_auc']:.4f}")
    
    return model, checkpoint_info


if __name__ == '__main__':
    main()
    
    # 使用示例：加载保存的模型进行推理
    # checkpoint_path = r"d:\Share\mobilenet_tsm\output\baseline\checkpoints\best_model_20251017_143025.pth"
    # model, info = load_model_for_inference(checkpoint_path, device='cuda')


```

训练轮数：36

验证集最佳性能：Loss: 0.4204,	ACC: 0.8955,	AUC: 0.9539,	F1: 0.8983

测试集：

ACC: 0.8807
AUC: 0.9558
F1:  0.8865