"""主入口脚本"""
import argparse
import warnings
import os
import sys

# 添加项目根目录到路径，支持绝对导入
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

def main():
    # 全局抑制第三方库冗余告警（优先在导入训练模块前设置）
    os.environ.setdefault('PYTHONWARNINGS', 'ignore')
    warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
    warnings.filterwarnings("ignore", message=".*frozen attribute.*")
    warnings.filterwarnings("ignore", message=".*repr attribute.*")
    warnings.filterwarnings("ignore", category=Warning, module=r"pydantic.*")

    # 延迟导入，确保告警过滤已生效
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='preprocess / train / eval')
    parser.add_argument('--config', type=str, default='configs/train_config.yml')
    args = parser.parse_args()

    if args.mode == 'preprocess':
        print('Running data preprocessing...')
    elif args.mode == 'train':
        from src.train.trainer import train_from_config
        print('Training MNTSM model...')
        train_from_config(args.config)
    elif args.mode == 'eval':
        from src.train.generalization_test import generalization_test
        print('Evaluating model performance (generalization test)...')
        generalization_test(args.config)
    else:
        raise ValueError('Unsupported mode')

if __name__ == '__main__':
    main()
