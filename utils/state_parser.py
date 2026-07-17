import numpy as np

class StateParser:
    """
    状态解析工具类
    负责将环境中致密的状态矩阵 (N, state_dim) 解析为结构化的数据，
    方便各种基线算法 (Baseline) 或自定义智能体提取特征。
    """
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        
        # 预定义状态在特征向量中的索引位置，需与 envs/liquid_airan_env.py 保持绝对同步
        self.idx_channel_gain = 0
        self.idx_compute_power = 1
        
        self.idx_label_dist_start = 2
        self.idx_label_dist_end = self.idx_label_dist_start + self.num_classes

    def parse_state(self, state_matrix: np.ndarray) -> dict:
        """
        将原始状态矩阵分解为易于理解的字典格式。
        
        Args:
            state_matrix (np.ndarray): 形状为 (N, 2 + num_classes) 的二维数组。
                                       N 是动态参与的车辆数量。
                                       
        Returns:
            dict: 包含不同物理含义张量的字典。
        """
        if not isinstance(state_matrix, np.ndarray) or state_matrix.ndim != 2:
            raise ValueError("State matrix must be a 2D numpy array with shape (N, state_dim).")
            
        N, state_dim = state_matrix.shape
        expected_dim = 2 + self.num_classes
        if state_dim != expected_dim:
            raise ValueError(f"Expected state dimension {expected_dim}, but got {state_dim}.")

        parsed = {
            'N': N,  # 当前活跃车辆数
            'channel_gains': state_matrix[:, self.idx_channel_gain],   # (N,) 信道增益
            'compute_powers': state_matrix[:, self.idx_compute_power], # (N,) 本地可用算力
            'label_distributions': state_matrix[:, self.idx_label_dist_start:self.idx_label_dist_end] # (N, num_classes)
        }
        return parsed

    def get_normalized_features(self, state_matrix: np.ndarray, feature_keys: list = None) -> np.ndarray:
        """
        [为协作者准备的工具] 提取特定的物理特征并进行简单的 Min-Max 归一化。
        这对于 MLP 或 K-Means 等对数值缩放敏感的算法极其重要。
        
        Args:
            state_matrix: 原始状态矩阵
            feature_keys: 需要提取的特征名列表，例如 ['channel_gains', 'compute_powers']
                          如果为 None，则提取除标签分布外的所有标量特征。
                          
        Returns:
            np.ndarray: 提取并列拼接后的归一化特征矩阵 (N, num_selected_features)
        """
        parsed = self.parse_state(state_matrix)
        
        if feature_keys is None:
            feature_keys = ['channel_gains', 'compute_powers']
            
        selected_features = []
        for key in feature_keys:
            if key not in parsed:
                raise KeyError(f"Feature '{key}' not found in parsed state. Available keys: {list(parsed.keys())}")
            
            feat = parsed[key]
            # 防御性编程：避免除以零
            val_min = np.min(feat)
            val_max = np.max(feat)
            if val_max > val_min:
                norm_feat = (feat - val_min) / (val_max - val_min)
            else:
                norm_feat = np.zeros_like(feat)
                
            selected_features.append(norm_feat.reshape(-1, 1))
            
        return np.concatenate(selected_features, axis=1)