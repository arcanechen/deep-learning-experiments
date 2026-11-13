"""Read saved epoch boundaries; does not start or modify training."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
plan=json.loads((root/'experiment_plan.json').read_text(encoding='utf-8'))
complete=0
for seed in plan['seeds']:
    for model in plan['models']:
        p=root/'runs'/f'{model}_s{seed}'/'progress.json'
        if p.exists():
            d=json.loads(p.read_text(encoding='utf-8'))
            complete+=int(d['complete'])
            print(f'{model}_s{seed}: {d["epoch"]}/{d["epochs"]}, best epoch={d["best_epoch"]}, val MSE={d["best_val_mse"]:.6f}')
        else:
            print(f'{model}_s{seed}: not started')
print(f'Completed training: {complete}/9')
print('Evaluation:', 'saved' if (root/'reports/evaluation/verification.json').exists() else 'not completed')
