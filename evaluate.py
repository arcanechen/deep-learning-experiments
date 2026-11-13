"""模型与训练设置确定后，对保存的最佳权重进行一次独立测试。"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data import prepare_test, preprocess, hwdb_metadata
from models import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    run_dir = args.run.resolve()
    if (run_dir / 'test_metrics.json').exists():
        raise FileExistsError('This run already has test results; read them instead of overwriting.')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    checkpoint = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
    use_amp = checkpoint.get('amp', False)
    if use_amp and device.type != 'cuda':
        raise ValueError('AMP baseline evaluation requires CUDA to preserve its inference precision.')
    if checkpoint['dataset'] == 'hwdb10':
        trained_data = json.loads((run_dir / 'data_info.json').read_text(encoding='utf-8'))
        current_data = hwdb_metadata()
        if (trained_data['manifest_sha256'] != current_data['manifest_sha256']
                or trained_data['classes'] != current_data['classes']
                or trained_data['array_sha256'] != current_data['array_sha256']):
            raise ValueError('HWDB dataset changed since training; refusing mismatched evaluation.')
    model = build_model(checkpoint['model']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    test, classes = prepare_test(checkpoint['dataset'])
    loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=0,
                        pin_memory=device.type == 'cuda')
    predictions, truths = [], []
    with torch.inference_mode():
        for images, labels in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                predicted = model(preprocess(images, device)).argmax(1)
            predictions.extend(predicted.cpu().tolist())
            truths.extend(labels.tolist())
    labels = list(range(10))
    matrix = confusion_matrix(truths, predictions, labels=labels)
    result = {'dataset': checkpoint['dataset'], 'model': checkpoint['model'],
              'checkpoint_epoch': checkpoint['epoch'], 'test_size': len(test),
              'accuracy': accuracy_score(truths, predictions),
              'macro_f1': f1_score(truths, predictions, labels=labels, average='macro', zero_division=0),
              'classification_report': classification_report(truths, predictions, labels=labels,
                    target_names=classes, output_dict=True, zero_division=0)}
    np.savetxt(run_dir / 'confusion_matrix.csv', matrix, delimiter=',', fmt='%d')
    with (run_dir / 'test_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['test_index', 'true_label', 'predicted_label'])
        writer.writerows(zip(range(len(test)), truths, predictions))
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    im = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(im, ax=ax, label='Number of images')
    ax.set(xticks=labels, yticks=labels, xticklabels=classes, yticklabels=classes,
           xlabel='Predicted class', ylabel='True class', title='Test confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    for i in labels:
        for j in labels:
            ax.text(j, i, matrix[i, j], ha='center', va='center', fontsize=8,
                    color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    fig.savefig(run_dir / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    wrong = np.flatnonzero(np.array(truths) != np.array(predictions))[:16]
    if len(wrong):
        fig, axes = plt.subplots(4, 4, figsize=(10, 10), constrained_layout=True)
        for ax in axes.flat:
            ax.axis('off')
        for ax, idx in zip(axes.flat, wrong):
            ax.imshow(test[int(idx)][0].squeeze().numpy(), cmap='gray')
            ax.set_title(f'True: {classes[truths[idx]]}\nPred: {classes[predictions[idx]]}', fontsize=8)
        fig.suptitle('First 16 errors in test order (not selected by appearance)')
        fig.savefig(run_dir / 'error_examples.png', dpi=160)
        plt.close(fig)
    (run_dir / 'test_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    summary_path = run_dir / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    summary['test_evaluated'] = True
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'classification_report'}, indent=2))


if __name__ == '__main__':
    main()
