"""Run real CUDA forward/backward updates and validate temporal data boundaries.

This is a pipeline check, NOT a trained model comparison or test-set evaluation.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from prepare_data import DATA, ROOT, save_json, valid_starts
from weather_model import WeatherWindows, WeatherForecaster


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    meta = json.loads((DATA / 'preprocessing.json').read_text(encoding='utf-8'))
    data = np.load(DATA / 'prepared.npz')
    values = data['values']
    # Independent brute-force missingness check over every eligible window.
    observed = np.isfinite(values).all(axis=1)
    for split, info in meta['splits'].items():
        expected = np.array([s for s in range(info['start_index'], info['end_index_exclusive']-335)
                             if observed[s:s+336].all()], dtype=np.int64)
        assert np.array_equal(data[split+'_starts'], expected)
    train_end = meta['splits']['train']['end_index_exclusive']
    train = values[:train_end][observed[:train_end]]
    np.testing.assert_allclose(train.mean(axis=0, dtype=np.float64), 0, atol=1e-5)
    np.testing.assert_allclose(train.std(axis=0, dtype=np.float64), 1, atol=1e-5)
    toy = np.ones((12, 2)); toy[5] = np.nan
    assert valid_starts(toy, 0, 12, width=4).tolist() == [0, 1, 6, 7, 8]
    dataset = WeatherWindows(values, data['train_starts'], meta['target_index'])
    x, y = next(iter(DataLoader(dataset, batch_size=32, shuffle=False)))
    assert x.shape == (32, 168, len(meta['features'])) and y.shape == (32, 168)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    for cell in ['rnn', 'gru', 'lstm']:
        torch.manual_seed(42)
        model = WeatherForecaster(cell, len(meta['features'])).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        before = model.head.weight.detach().clone()
        losses = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x.to(device))
            assert prediction.shape == y.shape
            loss = torch.nn.functional.mse_loss(prediction, y.to(device))
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        assert not torch.equal(before, model.head.weight)
        results[cell] = {'parameters': sum(p.numel() for p in model.parameters()), 'three_training_batch_losses': losses}
        print(cell, results[cell], flush=True)
    save_json(ROOT / 'reports' / 'pipeline_check.json', {
        'status': 'passed', 'scope': 'All window boundaries and missingness checked; three optimizer steps per model on one training batch; NOT formal training',
        'device': str(device), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
        'torch': str(torch.__version__), 'input_shape': list(x.shape), 'target_shape': list(y.shape), 'models': results,
        'prepared_sha256': meta['prepared_sha256']})


if __name__ == '__main__':
    main()
