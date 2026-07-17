import torch
import torch.nn as nn

class ResNet18_USFL(nn.Module):
    """
    支持 U 型拆分 (USFL) 的 ResNet-18 网络。
    将网络拆分为多个阶段，以便根据环境决策的 (l1, l2) 动态提取特征。
    """
    def __init__(self, num_classes=100):
        super(ResNet18_USFL, self).__init__()
        # 使用 torchvision 中预定义的 ResNet18 结构，但我们将其模块化
        from torchvision.models import resnet18
        # 这里为了演示通用性，保持原生结构，并将其划分为 7 个候选切分阶段
        original_model = resnet18(num_classes=num_classes)
        
        # 预先定义好各个切分阶段，构成一个 ModuleList
        self.layers = nn.ModuleList([
            nn.Sequential(original_model.conv1, original_model.bn1, original_model.relu, original_model.maxpool), # 阶段 0 (输入端)
            original_model.layer1, # 阶段 1
            original_model.layer2, # 阶段 2
            original_model.layer3, # 阶段 3
            original_model.layer4, # 阶段 4
            original_model.avgpool, # 阶段 5
            nn.Sequential(nn.Flatten(), original_model.fc) # 阶段 6 (输出端)
        ])
        self.num_layers = len(self.layers)

    def forward_partA(self, x, l1):
        """
        设备端前端执行 (Part A)。
        输入原始数据 x，执行到 l1 层，返回低维中间特征 (Smashed Data)。
        """
        for i in range(l1):
            x = self.layers[i](x)
        return x

    def forward_partB(self, smashed_data_batch, l1, l2):
        """
        边缘基站执行 (Part B)。
        接收物理拼接后的大 Batch Smashed Data，从 l1 执行到 l2 层。
        """
        x = smashed_data_batch
        for i in range(l1, l2):
            x = self.layers[i](x)
        return x

    def forward_partC(self, advanced_smashed_data, l2):
        """
        设备端后端执行 (Part C)。
        接收边缘返回的高阶特征，执行剩余层并计算 Loss。
        """
        x = advanced_smashed_data
        for i in range(l2, self.num_layers):
            x = self.layers[i](x)
        return x