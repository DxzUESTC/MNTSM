"""主入口脚本"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='preprocess / train / eval')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    args = parser.parse_args()

    if args.mode == 'preprocess':
        print('Running data preprocessing...')
    elif args.mode == 'train':
        print('Training MNTSM model...')
    elif args.mode == 'eval':
        print('Evaluating model performance...')
    else:
        raise ValueError('Unsupported mode')

if __name__ == '__main__':
    main()
