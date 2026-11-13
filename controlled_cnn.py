"""后续控制变量实验使用的同一家族 CNN，不改变五种基础模型。"""
from torch import nn


class ControlledCNN(nn.Module):
    """3 个阶段；每阶段 depth_per_stage 个卷积，通道依次 W/2W/4W。"""
    def __init__(self, depth_per_stage=2, width=32):
        super().__init__()
        if not isinstance(depth_per_stage, int) or depth_per_stage < 1:
            raise ValueError('depth_per_stage must be a positive integer')
        if not isinstance(width, int) or width < 1:
            raise ValueError('width must be a positive integer')
        self.depth_per_stage = depth_per_stage
        self.width = width
        layers = []
        in_channels = 1
        for out_channels in (width, 2 * width, 4 * width):
            for _ in range(depth_per_stage):
                layers.extend([nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                               nn.BatchNorm2d(out_channels), nn.ReLU()])
                in_channels = out_channels
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(4 * width, 10)

    def forward(self, images):
        features = self.features(images)
        # 对每个通道的所有空间位置取平均：[N,4W,4,4] -> [N,4W]。
        features = features.mean(dim=(2, 3))
        return self.classifier(features)


def expected_parameters(depth_per_stage, width):
    return ((90 + 189 * (depth_per_stage - 1)) * width**2
            + (63 + 14 * (depth_per_stage - 1)) * width + 10)
