"""固定方案完成训练后评估；预测先原子存盘，恢复时复用已保存预测。"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from ablation_common import atomic_json, exclusive_lock, write_csv
from controlled_cnn import ControlledCNN
from data import prepare_test, prepare_split, preprocess
from train import seed_everything


def evaluate(folder):
    with exclusive_lock(folder / 'run.lock'):
        if (folder / 'test_metrics.json').exists():
            print(f'REUSE TEST {folder.name}', flush=True)
            return
        torch.set_num_threads(4)
        seed_everything(42)
        if not (folder / 'summary.json').exists():
            raise ValueError('Training is incomplete')
        definition = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        _, _, current_data = prepare_split(definition['dataset'])
        saved_data = json.loads((folder / 'data_info.json').read_text(encoding='utf-8'))
        if saved_data != current_data:
            raise ValueError('Data split changed since training')
        checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        if checkpoint['definition'] != definition:
            raise ValueError('Checkpoint configuration mismatch')
        checkpoint_hash = hashlib.sha256((folder / 'best.pt').read_bytes()).hexdigest()
        cache = folder / 'test_prediction_cache.json'
        test, classes = prepare_test(definition['dataset'])
        if cache.exists():
            saved = json.loads(cache.read_text(encoding='utf-8'))
            if saved['checkpoint_sha256'] != checkpoint_hash:
                raise ValueError('Saved predictions belong to a different checkpoint')
            truth, pred = saved['truth'], saved['prediction']
        else:
            model = ControlledCNN(definition['depth_per_stage'], definition['width']).cuda()
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            truth, pred = [], []
            with torch.inference_mode():
                for images, labels in DataLoader(test, batch_size=256, shuffle=False, pin_memory=True):
                    with torch.autocast('cuda', dtype=torch.float16):
                        output = model(preprocess(images, torch.device('cuda')))
                    truth.extend(labels.tolist())
                    pred.extend(output.argmax(1).cpu().tolist())
            atomic_json(cache, {'checkpoint_sha256': checkpoint_hash, 'truth': truth, 'prediction': pred})
        assert len(truth) == len(pred) == len(test)
        assert truth == [int(test[i][1]) for i in range(len(test))]
        matrix = confusion_matrix(truth, pred, labels=list(range(10)))
        write_csv(folder / 'test_predictions.csv', [
            {'test_index': i, 'true_label': t, 'predicted_label': p} for i, (t, p) in enumerate(zip(truth, pred))])
        np.savetxt(folder / 'confusion_matrix.csv', matrix, delimiter=',', fmt='%d')
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        image = ax.imshow(matrix, cmap='Blues')
        fig.colorbar(image, ax=ax)
        ax.set(xticks=range(10), yticks=range(10), xticklabels=classes, yticklabels=classes,
               xlabel='Predicted', ylabel='True', title=folder.name)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        fig.savefig(folder / 'confusion_matrix.png', dpi=130)
        plt.close(fig)
        atomic_json(folder / 'test_metrics.json', {
            'accuracy': accuracy_score(truth, pred),
            'macro_f1': f1_score(truth, pred, average='macro', labels=list(range(10)), zero_division=0),
            'test_size': len(test), 'checkpoint_epoch': checkpoint['epoch'],
            'checkpoint_sha256': checkpoint_hash,
            'classification_report': classification_report(truth, pred, target_names=classes,
                                     labels=list(range(10)), output_dict=True, zero_division=0)})
        print(f'TEST {folder.name}: accuracy={accuracy_score(truth, pred):.4%}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    evaluate(parser.parse_args().run.resolve())
