"""Deterministic FP32 training, atomic epoch checkpoints, and resumable queue."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from prepare_data import ROOT, DATA
from weather_model import WeatherWindows, WeatherForecaster


def atomic(path, obj, tensor=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb' if tensor else 'w', **({} if tensor else {'encoding': 'utf-8'})) as f:
        if tensor:
            torch.save(obj, f)
        else:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def rng_state(generator):
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), *n[2:]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loader': generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader'])


def setup(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def epoch(model, loader, device, optimizer=None, clip=1.0):
    model.train(optimizer is not None)
    total = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = torch.nn.functional.mse_loss(prediction, y)
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Nonfinite loss; incomplete epoch will not be committed')
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
                optimizer.step()
            total += loss.detach().double() * y.numel()
            count += y.numel()
    return total.item() / count


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {name: sha(ROOT / name) for name in ['train_weather.py', 'weather_model.py', 'prepare_data.py', 'experiment_plan.json']}


def config_for(plan, cell, seed, meta, device):
    return {'plan': plan, 'model': cell, 'seed': seed, 'features': meta['features'],
            'target_index': meta['target_index'], 'prepared_sha256': meta['prepared_sha256'],
            'preprocessing_sha256': sha(DATA / 'preprocessing.json'), 'source_sha256': sources(),
            'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'numpy': np.__version__, 'precision': 'fp32_no_tf32', 'threads': 4, 'workers': 0}


def train_run(plan, cell, seed, run, data, meta, device, stop_after=None, honor_stop=True):
    with lock(run / 'run.lock'):
        cfg = config_for(plan, cell, seed, meta, device)
        if (run / 'config.json').exists():
            old = json.loads((run / 'config.json').read_text(encoding='utf-8'))
            if old != cfg:
                raise ValueError(f'Configuration/source/data/environment changed: {run}')
        else:
            atomic(run / 'config.json', cfg)
            for name in cfg['source_sha256']:
                dest = run / 'source' / name
                dest.parent.mkdir(exist_ok=True)
                dest.write_bytes((ROOT / name).read_bytes())
        setup(seed)
        generator = torch.Generator().manual_seed(seed)
        loaders = {}
        for split in ['train', 'val']:
            dataset = WeatherWindows(data['values'], data[split + '_starts'], meta['target_index'], plan['input_hours'], plan['horizon_hours'])
            loaders[split] = DataLoader(dataset, batch_size=plan['batch_size'], shuffle=split == 'train',
                                        generator=generator, num_workers=0, drop_last=False)
        model = WeatherForecaster(cell, len(meta['features']), plan['hidden_size'], plan['horizon_hours']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
        state = {'epoch': 0, 'best_val_mse': float('inf'), 'best_epoch': 0, 'history': [], 'best_model': None}
        checkpoint = run / 'last.pt'
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if state['config'] != cfg:
                raise ValueError('Checkpoint configuration mismatch')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            restore_rng(state['rng'], generator)
        while state['epoch'] < plan['epochs']:
            if honor_stop and (ROOT / 'STOP_AFTER_EPOCH').exists():
                break
            if stop_after is not None and state['epoch'] >= stop_after:
                break
            if device.type == 'cuda':
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            train_loss = epoch(model, loaders['train'], device, optimizer, plan['gradient_clip_norm'])
            val_loss = epoch(model, loaders['val'], device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            completed = state['epoch'] + 1
            record = {'epoch': completed, 'train_mse': train_loss, 'val_mse': val_loss,
                      'train_val_seconds': time.perf_counter()-start,
                      'peak_allocated_mb': torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None}
            if val_loss < state['best_val_mse']:
                state['best_val_mse'], state['best_epoch'] = val_loss, completed
                state['best_model'] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            state['history'].append(record)
            state.update(epoch=completed, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         rng=rng_state(generator), config=cfg)
            # One transaction contains both latest and best weights: cannot disagree after interruption.
            atomic(checkpoint, state, tensor=True)
            atomic(run / 'progress.json', {'epoch': completed, 'epochs': plan['epochs'], 'complete': completed == plan['epochs'],
                                          'best_epoch': state['best_epoch'], 'best_val_mse': state['best_val_mse'],
                                          'history': state['history']})
            print(f'{cell} s{seed} {completed}/{plan["epochs"]} train={train_loss:.6f} val={val_loss:.6f} {record["train_val_seconds"]:.1f}s', flush=True)
        return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=['rnn','gru','lstm'], default=['rnn','gru','lstm'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    plan = json.loads((ROOT / 'experiment_plan.json').read_text(encoding='utf-8'))
    if any(s not in plan['seeds'] for s in args.seeds):
        raise ValueError('Seed not in frozen plan')
    meta = json.loads((DATA / 'preprocessing.json').read_text(encoding='utf-8'))
    if sha(DATA / 'prepared.npz') != meta['prepared_sha256']:
        raise ValueError('Prepared data SHA-256 mismatch')
    data = dict(np.load(DATA / 'prepared.npz'))
    with lock(ROOT / 'runs' / 'queue.lock'):
        for seed in args.seeds:
            for cell in args.models:
                if (ROOT / 'STOP_AFTER_EPOCH').exists():
                    print('Stopped at saved epoch boundary. Remove weather/STOP_AFTER_EPOCH to continue.', flush=True)
                    return
                run = ROOT / 'runs' / f'{cell}_s{seed}'
                train_run(plan, cell, seed, run, data, meta, torch.device(args.device))
    print('Requested training queue complete. Test data has not been evaluated.', flush=True)


if __name__ == '__main__':
    main()
