"""CIFAR-100 ResNet-18 that supports U-shaped split-learning execution."""
import torch.nn as nn


class ResNet18_USFL(nn.Module):
    """CIFAR-optimized ResNet-18 with seven valid USFL split stages.

    GroupNorm avoids non-IID BatchNorm-buffer drift during FedAvg and remains
    well-defined for small per-client mini-batches.
    """

    def __init__(self, num_classes=100, group_norm_groups=8):
        super().__init__()
        from torchvision.models import resnet18

        def norm_layer(channels):
            return nn.GroupNorm(group_norm_groups, channels)

        model = resnet18(num_classes=num_classes, norm_layer=norm_layer)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        self.layers = nn.ModuleList([
            nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            nn.Sequential(nn.Flatten(), model.fc),
        ])
        self.num_layers = len(self.layers)

    def forward_partA(self, inputs, l1):
        for layer_index in range(l1):
            inputs = self.layers[layer_index](inputs)
        return inputs

    def forward_partB(self, smashed_data_batch, l1, l2):
        outputs = smashed_data_batch
        for layer_index in range(l1, l2):
            outputs = self.layers[layer_index](outputs)
        return outputs

    def forward_partC(self, advanced_smashed_data, l2):
        outputs = advanced_smashed_data
        for layer_index in range(l2, self.num_layers):
            outputs = self.layers[layer_index](outputs)
        return outputs
