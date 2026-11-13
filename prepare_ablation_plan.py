"""固定并核对控制变量配置，只生成实验清单，不启动训练。"""
import json
from pathlib import Path

from controlled_cnn import ControlledCNN, expected_parameters

ROOT = Path(__file__).resolve().parent


def build_plan():
    groups = {
        'depth': [(d, 32, 128) for d in (1, 2, 3)],
        'width': [(2, w, 128) for w in (16, 32, 64)],
        'batch': [(2, 32, b) for b in (32, 64, 128, 256)],
        'matched_parameters': [(1, 48, 128), (2, 27, 128), (3, 21, 128)],
    }
    unique = sorted({item for rows in groups.values() for item in rows})
    configs = []
    for depth, width, batch in unique:
        count = sum(p.numel() for p in ControlledCNN(depth, width).parameters())
        assert count == expected_parameters(depth, width)
        steps_per_epoch = (54000 + batch - 1) // batch
        configs.append({'id': f'd{depth}_w{width}_b{batch}',
                        'depth_per_stage': depth, 'conv_layers': 3 * depth,
                        'width': width, 'batch_size': batch, 'parameters': count,
                        'groups': [name for name, rows in groups.items() if (depth, width, batch) in rows],
                        'batches_per_epoch': steps_per_epoch,
                        'planned_optimizer_steps': 20 * steps_per_epoch})
    return {'status': 'PLANNED_NOT_TRAINED', 'dataset': 'fashion_mnist',
            'split_seed': 2026, 'train_size': 54000, 'val_size': 6000,
            'training_seeds': [42, 43, 44], 'epochs': 20, 'lr': 0.001,
            'optimizer': 'Adam', 'weight_decay': 0.0, 'precision': 'amp_float16',
            'selection': 'minimum validation cross-entropy',
            'unique_configurations': len(configs), 'planned_runs': len(configs) * 3,
            'notes': ['Shared configurations are reused across groups, not trained twice.',
                      'Fixed epochs do not imply fixed optimizer update counts.',
                      'AMP may skip optimizer updates on non-finite gradients; log actual steps when executed.',
                      'No test-based configuration selection.'], 'configurations': configs}


def main():
    plan = build_plan()
    destination = ROOT / 'reports' / 'ablation_plan.json'
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != plan:
            raise ValueError('Existing plan differs; review changes explicitly instead of overwriting.')
    else:
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(f'PLANNED ONLY: {plan["unique_configurations"]} configurations, {plan["planned_runs"]} runs')
    for config in plan['configurations']:
        print(config['id'], config['parameters'], 'parameters;', config['planned_optimizer_steps'], 'planned steps')


if __name__ == '__main__':
    main()
