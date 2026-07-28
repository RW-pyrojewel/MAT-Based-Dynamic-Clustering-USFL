import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class CIFAR100NonIIDProvider:
    """
    CIFAR-100 联邦/拆分学习非独立同分布 (Non-IID) 数据提供器。
    
    使用狄利克雷分布 Dirichlet(alpha) 对 CIFAR-100 进行划分，模拟设备间的数据异构。
    由于 CIFAR-100 包含 10 个类别，生成的标签分布向量 v_n 为 10 维。
    """
    def __init__(self, num_clients=25, alpha=1.0, data_dir="./data"):
        self.num_clients = num_clients
        self.alpha = alpha
        self.data_dir = data_dir
        self.num_classes = 100
        
        # 定义 CIFAR-100 标准预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        # 加载完整 CIFAR-100 训练集
        self.full_train_dataset = datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=self.transform
        )
        
        # 记录每个客户端分配到的样本索引列表
        self.client_indices = {i: [] for i in range(self.num_clients)}
        # 记录每个客户端的真实 10 维标签统计分布 v_n (用于 MAT-RL 状态输入)
        self.client_label_distributions = np.zeros((self.num_clients, self.num_classes))
        
        # 执行狄利克雷划分
        self._partition_data()

    def _partition_data(self):
        """
        核心物理逻辑：基于 Dirichlet 分布对数据索引进行 Non-IID 切分
        """
        # 1. 提取训练集中所有的标签
        targets = np.array(self.full_train_dataset.targets)
        
        # 2. 按类别收集所有样本的索引
        # class_indices[c] 包含所有属于类别 c 的样本在整个数据集中的位置
        class_indices = {c: np.where(targets == c)[0] for c in range(self.num_classes)}
        
        # 3. 遍历每个类别，将其分给各个客户端
        for c in range(self.num_classes):
            idx_list = class_indices[c]
            np.random.shuffle(idx_list) # 打乱该类别的样本顺序
            
            # 使用 Dirichlet(alpha) 产生 num_clients 维度的比例向量
            # 例如 alpha=0.1 时，比例向量会极度不均匀（有的人分极多，有的人分极少）
            proportions = np.random.dirichlet(np.ones(self.num_clients) * self.alpha)
            
            # 将比例转化为样本数量，并处理由于取整导致的尾数差异
            proportions = (proportions * len(idx_list)).astype(int)
            proportions[-1] = len(idx_list) - np.sum(proportions[:-1])
            
            # 根据计算出的数量切分索引，并分发给各客户端
            start = 0
            for client_id in range(self.num_clients):
                end = start + proportions[client_id]
                self.client_indices[client_id].extend(idx_list[start:end])
                
                # 记录该客户端在该类别下的样本数量
                self.client_label_distributions[client_id, c] = proportions[client_id]
                start = end

        # 4. 规范化标签分布统计向量 v_n，使其元素之和为 1 (转化为概率分布)
        for client_id in range(self.num_clients):
            np.random.shuffle(self.client_indices[client_id]) # 客户端内部打乱
            total_samples = np.sum(self.client_label_distributions[client_id])
            if total_samples > 0:
                self.client_label_distributions[client_id] /= total_samples
            else:
                # 兜底避免分到 0 个样本的极端情况
                self.client_label_distributions[client_id] = np.ones(self.num_classes) / self.num_classes

    def get_client_dataloader(self, client_id, batch_size=32, shuffle=True):
        """
        获取指定客户端对应的真实 PyTorch DataLoader
        """
        if client_id < 0 or client_id >= self.num_clients:
            raise ValueError(f"客户端 ID 必须在 0 到 {self.num_clients - 1} 之间")
            
        indices = self.client_indices[client_id]
        client_subset = Subset(self.full_train_dataset, indices)
        
        return DataLoader(
            client_subset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=False
        )

    def get_client_label_dist(self, client_id):
        """
        获取指定客户端的 10 维标签分布向量 v_n (符合概率分布，和为 1)
        """
        if client_id < 0 or client_id >= self.num_clients:
            raise ValueError(f"客户端 ID 必须在 0 到 {self.num_clients - 1} 之间")
            
        return self.client_label_distributions[client_id]