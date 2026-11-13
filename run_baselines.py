"""按预先固定的配方顺序执行基础实验；复用已完成的相同配置，禁止覆盖。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EPOCHS = {'mnist': 10, 'fashion_mnist': 10, 'hwdb10': 50}
MODELS = ('mlp', 'lenet', 'alexnet', 'googlenet', 'resnet18')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=tuple(EPOCHS), default=list(EPOCHS))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=['mlp', 'lenet'])
    parser.add_argument('--precision', choices=('fp32', 'amp'), default='fp32')
    args = parser.parse_args()
    for dataset in args.datasets:
        for model in args.models:
            suffix = '_amp' if args.precision == 'amp' else ''
            run_name = f'{dataset}_{model}_baseline{suffix}_s42'
            run_dir = ROOT / 'runs' / run_name
            epochs = EPOCHS[dataset]
            if run_dir.exists():
                config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
                expected = {'dataset': dataset, 'model': model, 'epochs': epochs,
                            'seed': 42, 'batch_size': 128, 'lr': 0.001, 'optimizer': 'Adam',
                            'precision': 'amp_float16' if args.precision == 'amp' else 'float32'}
                if any(config.get(k) != v for k, v in expected.items()):
                    raise ValueError(f'Existing run configuration mismatch: {run_name}')
                if not (run_dir / 'summary.json').exists():
                    raise ValueError(f'Incomplete existing run needs inspection: {run_name}')
                print(f'REUSE completed training: {run_name}', flush=True)
            else:
                print(f'TRAIN {run_name}, epochs={epochs}', flush=True)
                command = [sys.executable, '-u', str(ROOT / 'train.py'), '--dataset', dataset,
                                '--model', model, '--epochs', str(epochs), '--batch-size', '128',
                                '--lr', '0.001', '--seed', '42', '--run-name', run_name]
                if args.precision == 'amp':
                    command.append('--amp')
                subprocess.run(command, cwd=ROOT, check=True)
            if not (run_dir / 'test_metrics.json').exists():
                subprocess.run([sys.executable, '-u', str(ROOT / 'evaluate.py'), '--run', str(run_dir)], cwd=ROOT, check=True)
            else:
                print(f'REUSE existing test results: {run_name}', flush=True)
    print('REQUESTED_BASELINES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
