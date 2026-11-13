# 基础实验错误分析

以下由独立测试预测计算，用于描述固定实验结果。每个组合列出最多的三种误判方向，错误率以该真实类别的测试样本数为分母。原因仍需结合原图判断。

## mnist / mlp

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 7 - seven | 2 - two | 13 | 1.26% |
| 3 - three | 9 - nine | 13 | 1.29% |
| 5 - five | 3 - three | 8 | 0.90% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\mnist_mlp_baseline_s42`。

## mnist / lenet

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 5 - five | 3 - three | 9 | 1.01% |
| 9 - nine | 4 - four | 7 | 0.69% |
| 6 - six | 0 - zero | 7 | 0.73% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\mnist_lenet_baseline_s42`。

## mnist / alexnet

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 9 - nine | 8 - eight | 9 | 0.89% |
| 4 - four | 9 - nine | 9 | 0.92% |
| 5 - five | 3 - three | 7 | 0.78% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\mnist_alexnet_baseline_s42`。

## fashion_mnist / mlp

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| Shirt | T-shirt/top | 199 | 19.90% |
| Pullover | Coat | 118 | 11.80% |
| Shirt | Pullover | 107 | 10.70% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\fashion_mnist_mlp_baseline_s42`。

## fashion_mnist / lenet

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| T-shirt/top | Shirt | 150 | 15.00% |
| Shirt | T-shirt/top | 94 | 9.40% |
| Pullover | Shirt | 89 | 8.90% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\fashion_mnist_lenet_baseline_s42`。

## hwdb10 / mlp

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 中 | 木 | 7 | 11.67% |
| 水 | 木 | 6 | 10.17% |
| 木 | 水 | 6 | 10.00% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\hwdb10_mlp_baseline_s42`。

## hwdb10 / lenet

| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |
|---|---|---:|---:|
| 日 | 田 | 7 | 11.67% |
| 月 | 日 | 7 | 11.48% |
| 木 | 水 | 4 | 6.67% |

图片与完整混淆矩阵：`D:\深度学习作业\runs\hwdb10_lenet_baseline_s42`。

