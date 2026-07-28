"""Reproducible non-IID CIFAR-100 data provider for USFL experiments."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class CIFAR100NonIIDProvider:
    """Partition CIFAR-100 training samples across clients with Dirichlet(alpha)."""

    num_classes = 100

    def __init__(self, num_clients=30, alpha=0.1, data_dir="../Data", seed=7):
        if num_clients < 1 or alpha <= 0.0:
            raise ValueError("num_clients and alpha must be positive")
        self.num_clients = int(num_clients)
        self.alpha = float(alpha)
        self.data_dir = data_dir
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.full_train_dataset = datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = datasets.CIFAR100(
            root=self.data_dir, train=False, download=True, transform=self.test_transform
        )
        self.client_indices = {client_id: [] for client_id in range(self.num_clients)}
        self.client_label_distributions = np.zeros((self.num_clients, self.num_classes), dtype=np.float64)
        self._partition_data()

    def _partition_data(self):
        targets = np.asarray(self.full_train_dataset.targets)
        for class_id in range(self.num_classes):
            indices = np.flatnonzero(targets == class_id).copy()
            self.rng.shuffle(indices)
            proportions = self.rng.dirichlet(np.full(self.num_clients, self.alpha))
            counts = self.rng.multinomial(len(indices), proportions)
            offset = 0
            for client_id, count in enumerate(counts):
                self.client_indices[client_id].extend(indices[offset:offset + count].tolist())
                self.client_label_distributions[client_id, class_id] = count
                offset += count

        for client_id, indices in self.client_indices.items():
            self.rng.shuffle(indices)
            total = self.client_label_distributions[client_id].sum()
            if total > 0.0:
                self.client_label_distributions[client_id] /= total
            else:
                self.client_label_distributions[client_id] = 1.0 / self.num_classes

    def get_client_dataloader(self, client_id, batch_size=32, shuffle=True, drop_last=False):
        if not 0 <= int(client_id) < self.num_clients:
            raise ValueError("client_id is outside the configured client range")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = self.client_indices[int(client_id)]
        if not indices:
            raise RuntimeError(f"client {client_id} has no assigned CIFAR-100 samples")
        return DataLoader(
            Subset(self.full_train_dataset, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def get_train_dataloader(self, batch_size=128, shuffle=True):
        return DataLoader(
            self.full_train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def get_test_dataloader(self, batch_size=128):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def get_client_label_dist(self, client_id):
        if not 0 <= int(client_id) < self.num_clients:
            raise ValueError("client_id is outside the configured client range")
        return self.client_label_distributions[int(client_id)].copy()
