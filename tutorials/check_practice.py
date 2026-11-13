"""Recompute metrics of ONE saved GRU prediction batch, not the full validation set."""
from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parent/'runs'
folders = sorted(root.glob('gru_*'))
if not folders:
    raise FileNotFoundError('Run train_practice.py --task gru first')
folder = folders[-1]
config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
with np.load(folder/'example_prediction.npz') as data:
    prediction, target = data['prediction'], data['target']
assert prediction.shape == target.shape
scale = config['data']['target_std']
offset = config['data']['target_mean']
pred_degC = prediction*scale + offset
true_degC = target*scale + offset
error = pred_degC - true_degC
print('source:', folder)
print('shape:', error.shape)
print('MAE degC:', np.abs(error).mean())
print('RMSE degC:', np.sqrt(np.square(error).mean()))
print('RMSE of each horizon:', np.sqrt(np.square(error).mean(axis=0)))
