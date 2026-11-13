# 基础实验错误分析

以下由独立测试预测计算，用于描述固定实验结果。每个组合列出最多的三种误判方向，错误率以该真实类别的测试样本数为分母。原因仍需结合原图判断。

## mnist / mlp

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 4 - four | 9 - nine | 20 | 2.04% |
| 7 - seven | 9 - nine | 15 | 1.46% |
| 5 - five | 3 - three | 12 | 1.35% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\mnist_mlp_baseline_amp_s42`。

## mnist / lenet

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 7 - seven | 2 - two | 9 | 0.88% |
| 9 - nine | 4 - four | 9 | 0.89% |
| 5 - five | 3 - three | 8 | 0.90% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\mnist_lenet_baseline_amp_s42`。

## mnist / alexnet

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 9 - nine | 4 - four | 7 | 0.69% |
| 8 - eight | 9 - nine | 5 | 0.51% |
| 9 - nine | 7 - seven | 4 | 0.40% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\mnist_alexnet_baseline_amp_s42`。

## mnist / googlenet

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 5 - five | 3 - three | 10 | 1.12% |
| 4 - four | 9 - nine | 7 | 0.71% |
| 1 - one | 6 - six | 5 | 0.44% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\mnist_googlenet_baseline_amp_s42`。

## mnist / resnet18

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 4 - four | 9 - nine | 16 | 1.63% |
| 2 - two | 7 - seven | 6 | 0.58% |
| 4 - four | 6 - six | 6 | 0.61% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\mnist_resnet18_baseline_amp_s42`。

## fashion_mnist / mlp

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| Shirt | T-shirt/top | 114 | 11.40% |
| T-shirt/top | Shirt | 111 | 11.10% |
| Shirt | Pullover | 106 | 10.60% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\fashion_mnist_mlp_baseline_amp_s42`。

## fashion_mnist / lenet

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| T-shirt/top | Shirt | 130 | 13.00% |
| Shirt | Coat | 109 | 10.90% |
| Shirt | T-shirt/top | 101 | 10.10% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\fashion_mnist_lenet_baseline_amp_s42`。

## fashion_mnist / alexnet

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| T-shirt/top | Shirt | 130 | 13.00% |
| Shirt | Coat | 82 | 8.20% |
| Shirt | T-shirt/top | 78 | 7.80% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\fashion_mnist_alexnet_baseline_amp_s42`。

## fashion_mnist / googlenet

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| T-shirt/top | Shirt | 118 | 11.80% |
| Shirt | Coat | 109 | 10.90% |
| Shirt | T-shirt/top | 82 | 8.20% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\fashion_mnist_googlenet_baseline_amp_s42`。

## fashion_mnist / resnet18

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| Shirt | T-shirt/top | 141 | 14.10% |
| Shirt | Pullover | 64 | 6.40% |
| Shirt | Coat | 63 | 6.30% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\fashion_mnist_resnet18_baseline_amp_s42`。

## hwdb10 / mlp

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 月 | 日 | 9 | 14.75% |
| 田 | 日 | 7 | 12.07% |
| 木 | 中 | 6 | 10.00% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\hwdb10_mlp_baseline_amp_s42`。

## hwdb10 / lenet

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 日 | 月 | 6 | 10.00% |
| 日 | 田 | 5 | 8.33% |
| 木 | 水 | 4 | 6.67% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\hwdb10_lenet_baseline_amp_s42`。

## hwdb10 / alexnet

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 田 | 日 | 6 | 10.34% |
| 月 | 日 | 6 | 9.84% |
| 日 | 田 | 3 | 5.00% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\hwdb10_alexnet_baseline_amp_s42`。

## hwdb10 / googlenet

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 月 | 日 | 4 | 6.56% |
| 日 | 月 | 2 | 3.33% |
| 中 | 田 | 1 | 1.67% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\hwdb10_googlenet_baseline_amp_s42`。

## hwdb10 / resnet18

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 月 | 日 | 3 | 4.92% |
| 日 | 田 | 3 | 5.00% |
| 日 | 月 | 3 | 5.00% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\hwdb10_resnet18_baseline_amp_s42`。

