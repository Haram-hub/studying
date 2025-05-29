import os
import argparse
import sys
sys.path.append('Anomaly-Transformer-main')
from main import main

def run_experiment(dataset_name):
    # 데이터셋별 설정
    dataset_configs = {
        'SMD': {'input_c': 38, 'data_path': 'dataset/SMD'},
        'MSL': {'input_c': 55, 'data_path': 'dataset/MSL'},
        'SMAP': {'input_c': 25, 'data_path': 'dataset/SMAP'},
        'PSM': {'input_c': 25, 'data_path': 'dataset/PSM'}
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"지원하지 않는 데이터셋입니다. 사용 가능한 데이터셋: {list(dataset_configs.keys())}")
    
    config = dataset_configs[dataset_name]
    
    # 학습 실행
    print(f"\n=== {dataset_name} 데이터셋 학습 시작 ===")
    train_args = argparse.Namespace(
        anormly_ratio=0.5,
        num_epochs=10,
        batch_size=256,
        mode='train',
        dataset=dataset_name,
        data_path=config['data_path'],
        input_c=config['input_c'],
        pretrained_model=None,
        model_save_path='checkpoints'
    )
    main(train_args)
    
    # 테스트 실행
    print(f"\n=== {dataset_name} 데이터셋 테스트 시작 ===")
    test_args = argparse.Namespace(
        anormly_ratio=0.5,
        num_epochs=10,
        batch_size=256,
        mode='test',
        dataset=dataset_name,
        data_path=config['data_path'],
        input_c=config['input_c'],
        pretrained_model='checkpoints/best_model.pth',
        model_save_path='checkpoints'
    )
    main(test_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SMD',
                      choices=['SMD', 'MSL', 'SMAP', 'PSM'],
                      help='실행할 데이터셋 선택 (SMD, MSL, SMAP, PSM)')
    
    args = parser.parse_args()
    run_experiment(args.dataset) 