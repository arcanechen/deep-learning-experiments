"""实验一模型。统一输入 [N, 1, 32, 32]，输出 [N, 10] 原始分数。

大型网络仅适配灰度小图的入口与输出，保留主要网络结构。
"""
from torch import nn
from torchvision import models


class LeNet(nn.Module):
    """LeNet 风格版本：用 ReLU、最大池化替代历史版本的激活/池化。"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


MODEL_NAMES = ("mlp", "lenet", "alexnet", "googlenet", "resnet18")


def build_model(name):
    if name == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(32 * 32, 256), nn.ReLU(),
                             nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    if name == "lenet":
        return LeNet()
    if name == "alexnet":
        model = models.alexnet(weights=None, num_classes=10)
        model.features[0] = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        model.features[2] = nn.Identity()  # 小图不在入口再次下采样
        # 固定 32x32 输入时，特征图为 7x7；自适应 7->6 的所有窗口均为 2x2、步长1。
        # 普通平均池化数学等价，并支持 CUDA 严格确定性反向传播。
        model.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)
        return model
    if name == "googlenet":
        model = models.googlenet(weights=None, num_classes=10, aux_logits=False,
                                init_weights=False, transform_input=False)
        model.conv1.conv = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool1 = nn.Identity()
        model.maxpool2 = nn.Identity()
        return model
    if name == "resnet18":
        model = models.resnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    raise ValueError(f"Unknown model: {name}")
