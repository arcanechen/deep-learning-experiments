"""入门练习：张量、自动求导、LeNet 形状和一个训练 batch。

只读训练集，使用 CPU；不改动正式实验权重，不访问测试集。
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import ROOT, prepare_split, preprocess
from models import LeNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=40)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    device = torch.device('cpu')

    # 1. tensor 是带有 shape、dtype、device 等信息的多维数组。
    a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    print('1. TENSOR:', a, 'shape=', tuple(a.shape),
          'dtype=', a.dtype, 'device=', a.device)

    # 2. 对比手算梯度：L=(3w-12)^2，dL/dw=2(3w-12)*3。
    w = torch.tensor(2., requires_grad=True)
    loss = (w * 3 - 12).square()
    loss.backward()
    gradient = w.grad.item()
    with torch.no_grad():
        w -= 0.05 * w.grad
    print(f'2. AUTOGRAD: loss_before={loss.item():.2f}, gradient={gradient:.2f}, '
          f'w_after={w.item():.2f}, loss_after={(w * 3 - 12).square().item():.2f}')

    # 3. 固定取一个训练 batch。这里的 32 是图片数，不是图像宽度。
    train_data, _, _ = prepare_split('fashion_mnist')
    loader = DataLoader(train_data, batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(2026))
    raw_images, labels = next(iter(loader))
    images = preprocess(raw_images, device)
    print(f'3. BATCH: raw={tuple(raw_images.shape)} {raw_images.dtype}; '
          f'input={tuple(images.shape)} {images.dtype}; '
          f'labels={tuple(labels.shape)} {labels.dtype}; '
          f'input_range=[{images.min().item():.1f}, {images.max().item():.1f}]')

    # 4. 使用与正式实验相同的 LeNet，逐层打印形状。
    model = LeNet().to(device)
    shapes = []
    with torch.no_grad():
        x = images
        for group_name in ('features', 'classifier'):
            for index, layer in enumerate(getattr(model, group_name)):
                x = layer(x)
                row = {'layer': f'{group_name}.{index}: {type(layer).__name__}',
                       'shape': list(x.shape)}
                shapes.append(row)
                print('4. SHAPE:', row['layer'], '->', tuple(x.shape))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    def measure():
        # eval() 控制 Dropout/BN；no_grad() 关闭梯度记录，作用不同。
        model.eval()
        with torch.no_grad():
            logits = model(images)
            return {'loss': criterion(logits, labels).item(),
                    'accuracy': (logits.argmax(dim=1) == labels).float().mean().item()}

    before = measure()
    history = []
    first_weight_change = None
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)  # 清除旧梯度
        logits = model(images)                # [32,1,32,32] -> [32,10]
        loss = criterion(logits, labels)      # 原始分数 + 整数标签 -> 一个数
        loss.backward()                       # 计算梯度；此时还未修改参数
        if step == 1:
            saved_weight = model.features[0].weight.detach().clone()
            print(f'5. BACKWARD: first_conv_grad_norm='
                  f'{model.features[0].weight.grad.norm().item():.6f}')
        optimizer.step()                     # 根据梯度修改模型参数
        if step == 1:
            first_weight_change = (model.features[0].weight.detach() - saved_weight).abs().max().item()
            print(f'5. UPDATE: first_conv_max_weight_change={first_weight_change:.6f}')
        measured = measure()
        history.append({'step': step, **measured})
        if step in (1, 2, 5, 10, 20, args.steps):
            print(f'6. SAME BATCH: step={step:02d}, loss={measured["loss"]:.4f}, '
                  f'accuracy={measured["accuracy"]:.2%}')

    model.eval()
    with torch.no_grad():
        logits = model(images)
        probabilities = logits.softmax(dim=1)
        prediction = logits.argmax(dim=1)
    print('7. PREDICTION: first_8_true=', labels[:8].tolist(),
          'first_8_predicted=', prediction[:8].tolist())
    print('7. SOFTMAX: first_image_probability_sum=', probabilities[0].sum().item())
    print('This exercise repeatedly fits ONE training batch; it does NOT measure generalization.')
    out = ROOT / 'reports' / 'learning'
    out.mkdir(parents=True, exist_ok=True)
    result = {'purpose': 'CPU teaching exercise on one training batch; not a baseline or test result',
              'seed': 2026, 'batch_size': 32, 'steps': args.steps,
              'before': before, 'after': history[-1], 'history': history,
              'autograd_scalar_gradient': gradient,
              'first_conv_max_weight_change': first_weight_change,
              'layer_shapes': shapes,
              'first_image': {'true_label': labels[0].item(),
                              'predicted_label': prediction[0].item(),
                              'probabilities': probabilities[0].tolist()}}
    (out / 'one_batch_demo.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    names = ['T恤', '裤子', '套头衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴']
    axes[0, 0].imshow((images[0, 0] + 1) / 2, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[0, 0].set_title(f'一张输入图片：1×32×32\n真实类别：{names[labels[0].item()]}')
    axes[0, 0].axis('off')
    colors = ['#267e67' if i == labels[0].item() else '#96aabe' for i in range(10)]
    axes[0, 1].bar(names, probabilities[0].tolist(), color=colors)
    axes[0, 1].set(title=f'{args.steps} 步后：十个分数经 Softmax 转成概率', ylabel='概率', ylim=(0, 1.05))
    axes[0, 1].tick_params(axis='x', rotation=35)
    steps = [0] + [r['step'] for r in history]
    axes[1, 0].plot(steps, [before['loss']] + [r['loss'] for r in history], color='#2f6b9a')
    axes[1, 0].set(title='这 32 张图片的交叉熵损失', xlabel='参数更新次数', ylabel='损失')
    axes[1, 1].plot(steps, [before['accuracy'] * 100] + [r['accuracy'] * 100 for r in history], color='#267e67')
    axes[1, 1].set(title='这 32 张图片的分类准确率', xlabel='参数更新次数', ylabel='准确率（%）', ylim=(0, 105))
    for ax in axes[1]:
        ax.grid(alpha=0.2)
    fig.suptitle('教学练习：重复学习同一批训练图片\n本图不表示验证集或测试集成绩', fontsize=15)
    fig.savefig(out / 'learning_walkthrough.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
