"""主入口脚本"""
import argparse
import warnings
import os

def main():
    # 全局抑制第三方库冗余告警（优先在导入训练模块前设置）
    os.environ.setdefault('PYTHONWARNINGS', 'ignore')
    warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
    warnings.filterwarnings("ignore", message=".*frozen attribute.*")
    warnings.filterwarnings("ignore", message=".*repr attribute.*")
    warnings.filterwarnings("ignore", category=Warning, module=r"pydantic.*")

    # 延迟导入，确保告警过滤已生效
    from .train.trainer import train_from_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='preprocess / train / eval')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    args = parser.parse_args()

    if args.mode == 'preprocess':
        print('Running data preprocessing...')
    elif args.mode == 'train':
        print('Training MNTSM model...')
        train_from_config(args.config)
    elif args.mode == 'eval':
        print('Evaluating model performance...')
    else:
        raise ValueError('Unsupported mode')

if __name__ == '__main__':
    main()
