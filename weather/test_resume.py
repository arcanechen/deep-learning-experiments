"""Compare uninterrupted vs saved/reloaded training on deterministic synthetic data."""
import tempfile
from pathlib import Path
import json
import numpy as np
from train_weather import ROOT, DATA, train_run, torch, lock, atomic


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def main():
    plan = json.loads((ROOT / 'experiment_plan.json').read_text(encoding='utf-8'))
    plan.update(epochs=2, batch_size=8, hidden_size=8)
    meta = json.loads((DATA / 'preprocessing.json').read_text(encoding='utf-8'))
    data = {'values': np.random.default_rng(89).normal(size=(1100,19)).astype(np.float32),
            'train_starts': np.arange(32), 'val_starts': np.arange(600,616)}
    meta['prepared_sha256'] = 'synthetic_resume_test_only'
    results = []
    with tempfile.TemporaryDirectory(prefix='weather_resume_') as temp:
        base = Path(temp)
        for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
            for cell in ['rnn','gru','lstm']:
                run1, run2 = base/f'{device}_{cell}_full', base/f'{device}_{cell}_resume'
                train_run(plan, cell, 42, run1, data, meta, torch.device(device), honor_stop=False)
                train_run(plan, cell, 42, run2, data, meta, torch.device(device), stop_after=1, honor_stop=False)
                train_run(plan, cell, 42, run2, data, meta, torch.device(device), honor_stop=False)
                a = torch.load(run1/'last.pt', weights_only=True, map_location='cpu')
                b = torch.load(run2/'last.pt', weights_only=True, map_location='cpu')
                for key in ['model','best_model','optimizer','rng','epoch','best_epoch','best_val_mse']:
                    equal(a[key], b[key])
                for x,y in zip(a['history'], b['history']):
                    equal([x['train_mse'], x['val_mse']], [y['train_mse'], y['val_mse']])
                changed = dict(plan, learning_rate=.01)
                try:
                    train_run(changed, cell, 42, run2, data, meta, torch.device(device), honor_stop=False)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Changed configuration accepted')
                results.append({'device':device,'model':cell,'bitwise_resume':'passed','config_guard':'passed'})
        with lock(base/'guard.lock'):
            try:
                with lock(base/'guard.lock'):
                    raise AssertionError('Duplicate lock accepted')
            except OSError:
                pass
    atomic(ROOT/'reports/resume_verification.json', {'status':'passed','lock_guard':'passed','checks':results,
           'scope':'2 epochs synthetic data, same-process model reconstruction + disk reload; exact weights/optimizer/RNG/loss match; not cross-device reproducibility'})
    print('All resume and guard checks PASSED', flush=True)


if __name__ == '__main__':
    main()
