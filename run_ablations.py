"""单队列自动续训：先完成预设的全部训练，再统一独立测试与汇总。"""
import json
import subprocess
import sys

from ablation_common import ROOT, OUT, PLAN_PATH, exclusive_lock, run_name
from ablation_status import refresh


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(OUT / 'queue.lock'):
        plan = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
        jobs = [(c, seed) for seed in plan['training_seeds'] for c in plan['configurations']]
        refresh('training')
        for c, seed in jobs:
            if (OUT / 'STOP_AFTER_EPOCH').exists():
                refresh('paused_at_saved_epoch')
                return 75
            print(f'TRAIN/RESUME {c["id"]} seed={seed}', flush=True)
            result = subprocess.run([sys.executable, '-u', str(ROOT / 'train_ablation.py'),
                                     '--config', c['id'], '--seed', str(seed)], cwd=ROOT)
            if result.returncode:
                refresh('paused_at_saved_epoch' if result.returncode == 75 else 'training_error_saved_progress')
                return result.returncode
        refresh('evaluating')
        for c, seed in jobs:
            if (OUT / 'STOP_AFTER_EPOCH').exists():
                refresh('paused_before_test')
                return 75
            result = subprocess.run([sys.executable, '-u', str(ROOT / 'evaluate_ablation.py'),
                                     '--run', str(ROOT / 'runs' / run_name(c['id'], seed))], cwd=ROOT)
            if result.returncode:
                refresh('evaluation_error_saved_progress')
                return result.returncode
            refresh('evaluating')
        subprocess.run([sys.executable, str(ROOT / 'summarize_ablations.py')], cwd=ROOT, check=True)
        refresh('complete')
        print('ALL_33_ABLATIONS_COMPLETE', flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
