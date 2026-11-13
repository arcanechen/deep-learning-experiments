"""按固定清单训练单组控制实验，自动从最后完整轮次恢复。"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torchvision
from torch.utils.data import DataLoader

from ablation_common import (ROOT, OUT, PLAN_PATH, atomic_json, atomic_checkpoint,
                             exclusive_lock, epoch, rng_state, restore_rng, write_csv, run_name)
from ablation_status import refresh
from controlled_cnn import ControlledCNN
from data import prepare_split
from train import seed_everything, plot_history


def train(config_id, seed, stop_after_epoch=None):
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the frozen AMP experiment plan')
    plan = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
    config = next(c for c in plan['configurations'] if c['id'] == config_id)
    if seed not in plan['training_seeds']:
        raise ValueError('Seed not in frozen plan')
    folder = ROOT / 'runs' / run_name(config_id, seed)
    folder.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(folder / 'run.lock'):
        files = ['train_ablation.py', 'ablation_common.py', 'controlled_cnn.py', 'data.py', 'train.py']
        definition = {**config, 'seed': seed, 'dataset': plan['dataset'], 'epochs': plan['epochs'],
                      'lr': plan['lr'], 'precision': plan['precision'], 'optimizer': plan['optimizer'],
                      'weight_decay': plan['weight_decay'], 'python': platform.python_version(),
                      'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                      'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                      'plan_sha256': hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(),
                      'source_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}}
        config_path = folder / 'config.json'
        if config_path.exists():
            if json.loads(config_path.read_text(encoding='utf-8')) != definition:
                raise ValueError(f'Configuration/source/environment changed: {folder.name}')
        else:
            atomic_json(config_path, definition)
        source = folder / 'source'
        source.mkdir(exist_ok=True)
        for name in files:
            if not (source / name).exists():
                shutil.copy2(ROOT / name, source / name)
        if (folder / 'summary.json').exists():
            print(f'REUSE {folder.name}', flush=True)
            return 0
        seed_everything(seed)
        device = torch.device('cuda')
        train_data, val_data, info = prepare_split(plan['dataset'])
        data_path = folder / 'data_info.json'
        if data_path.exists() and json.loads(data_path.read_text(encoding='utf-8')) != info:
            raise ValueError('Data split changed')
        atomic_json(data_path, info)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                            generator=generator, pin_memory=True, num_workers=0)
        validation = DataLoader(val_data, batch_size=256, shuffle=False, pin_memory=True)
        model = ControlledCNN(config['depth_per_stage'], config['width']).to(device)
        assert sum(p.numel() for p in model.parameters()) == config['parameters']
        optimizer = torch.optim.Adam(model.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
        scaler = torch.amp.GradScaler('cuda')
        last_path = folder / 'last.pt'
        state = {'epoch': 0, 'history': [], 'best_loss': float('inf'), 'best_epoch': 0,
                 'best_state': None, 'compute_seconds': 0., 'peak_allocated_mb': 0.}
        if last_path.exists():
            state = torch.load(last_path, map_location='cpu', weights_only=True)
            if state['definition'] != definition:
                raise ValueError('Checkpoint definition mismatch')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scaler.load_state_dict(state['scaler'])
            restore_rng(state['rng'], generator)
            print(f'RESUME {folder.name} after epoch {state["epoch"]}', flush=True)
        torch.cuda.reset_peak_memory_stats()

        def commit():
            state.update({'definition': definition, 'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                          'rng': rng_state(generator)})
            atomic_checkpoint(last_path, state)
            atomic_json(folder / 'progress.json', {'epoch': state['epoch'], 'total_epochs': plan['epochs'],
                        'best_epoch': state['best_epoch'], 'best_val_loss': state['best_loss'] if state['epoch'] else None})
            if state['history']:
                write_csv(folder / 'history.csv', state['history'])
            refresh('training')

        if not last_path.exists():
            commit()
        for number in range(state['epoch'] + 1, plan['epochs'] + 1):
            if (OUT / 'STOP_AFTER_EPOCH').exists():
                refresh('paused_at_saved_epoch')
                return 75
            torch.cuda.synchronize()
            started = time.perf_counter()
            training = epoch(model, loader, device, optimizer, scaler)
            val = epoch(model, validation, device)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            row = {'epoch': number, 'train_loss': training['loss'], 'train_accuracy': training['accuracy'],
                   'val_loss': val['loss'], 'val_accuracy': val['accuracy'], 'seconds': seconds,
                   'attempted_updates': training['attempted_updates'], 'actual_updates': training['actual_updates']}
            state['history'].append(row)
            state['epoch'] = number
            state['compute_seconds'] += seconds
            state['peak_allocated_mb'] = max(state['peak_allocated_mb'], torch.cuda.max_memory_allocated() / 2**20)
            if val['loss'] < state['best_loss']:
                state['best_loss'], state['best_epoch'] = val['loss'], number
                state['best_state'] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            commit()
            print(f'{folder.name} {number:02d}/20 train={training["accuracy"]:.2%} '
                  f'val={val["accuracy"]:.2%} val_loss={val["loss"]:.4f} '
                  f'updates={training["actual_updates"]}/{training["attempted_updates"]} seconds={seconds:.1f}', flush=True)
            if stop_after_epoch is not None and number >= stop_after_epoch:
                return 75
        atomic_checkpoint(folder / 'best.pt', {'state_dict': state['best_state'], 'definition': definition,
                          'epoch': state['best_epoch'], 'val_loss': state['best_loss']})
        write_csv(folder / 'history.csv', state['history'])
        plot_history(state['history'], folder / 'learning_curves.png')
        atomic_json(folder / 'summary.json', {'best_epoch': state['best_epoch'], 'best_val_loss': state['best_loss'],
                    'best_val_accuracy': state['history'][state['best_epoch'] - 1]['val_accuracy'],
                    'parameters': config['parameters'], 'compute_seconds': state['compute_seconds'],
                    'timing_scope': 'successful epochs: training + validation; checkpoint IO and interrupted partial epochs excluded',
                    'peak_allocated_mb': state['peak_allocated_mb'],
                    'attempted_updates': sum(r['attempted_updates'] for r in state['history']),
                    'actual_updates': sum(r['actual_updates'] for r in state['history'])})
        refresh('training')
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--stop-after-epoch', type=int, help='Save and exit at a completed epoch; for recovery verification')
    args = parser.parse_args()
    raise SystemExit(train(args.config, args.seed, args.stop_after_epoch))
