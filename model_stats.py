"""统计实际 32×32 模型的参数量与 Conv/Linear MACs，不估算未计入算子。"""
import json
from pathlib import Path

import torch
from torch import nn

from models import MODEL_NAMES, build_model


def main():
    torch.set_num_threads(2)
    rows = []
    for name in MODEL_NAMES:
        model = build_model(name).eval()
        macs = [0]
        handles = []

        def count(module, inputs, output):
            if isinstance(module, nn.Conv2d):
                macs[0] += output.numel() * module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
            elif isinstance(module, nn.Linear):
                macs[0] += output.numel() * module.in_features

        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(count))
        with torch.inference_mode():
            model(torch.zeros(1, 1, 32, 32))
        for handle in handles:
            handle.remove()
        row = {'model': name, 'parameters': sum(p.numel() for p in model.parameters()),
               'conv_linear_macs_per_image': int(macs[0])}
        rows.append(row)
    out = Path(__file__).resolve().parent / 'reports'
    out.mkdir(exist_ok=True)
    result = {'input': [1, 1, 32, 32], 'scope': 'Conv2d and Linear MACs only; excludes BN, activation, pooling, bias, backward and optimizer; not measured latency', 'models': rows}
    (out / 'model_stats.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
