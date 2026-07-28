import numpy as np

class LiquidAIRANEnv:
    """
    6G AI-RAN 纯无线与系统状态环境
    职责边界:
    - 不模拟任何 GPU 计算耗时与特征大小（交由 PyTorch 在真实单卡硬件上结合逻辑时钟测定）。
    - 仅专注控制 150 轮环境潮汐（动态参与、带宽拥塞）。
    - 仅负责香农定理下的无线信道传输延迟计算。
    """
    def __init__(self, data_provider, max_vehicles=25, base_bandwidth=100e6, max_migs=5):
        """
        必须在初始化时传入实例化的 CIFAR10NonIIDProvider，以获取真实的 v_n
        """
        self.data_provider = data_provider
        self.max_vehicles = max_vehicles
        self.num_classes = data_provider.num_classes # 必须为 10
        self.base_bandwidth = base_bandwidth
        self.max_migs = max_migs
        if self.max_migs < 5:
            raise ValueError("max_migs must support the five-MIG tide phase")
        
        self.current_epoch = 0
        self.max_epochs = 150
        
        # 潮汐状态变量
        self.current_N = 0
        self.current_migs = 2
        self.current_bandwidth = self.base_bandwidth
        self.active_vehicle_ids = [] # 当前在场的车辆全局 ID 列表
        
        # ========================================================
        # 【持久化车辆池登记】 (Vehicle Registry)
        # ========================================================
        # 为每一辆可能参与联邦的车辆，生成并固化它的基础物理画像
        # 1. 固化的算力天赋 f_n (U(1.0, 2.5) GHz)
        self.vehicle_pool_computes = np.random.uniform(1.0, 2.5, size=self.max_vehicles)
        
        # 2. 绝对真实的标签分布 v_n (从 Provider 获取)
        self.vehicle_pool_labels = np.zeros((self.max_vehicles, self.num_classes))
        for vid in range(self.max_vehicles):
            # 获取这辆车真实的、用于 PyTorch 训练的数据集标签分布
            self.vehicle_pool_labels[vid] = self.data_provider.get_client_label_dist(vid)
        
    def reset(self):
        """初始化环境到第 0 轮"""
        self.current_epoch = 0
        return self._step_tide()
        
    def step(self):
        """
        环境演进，严格执行 150 轮三阶段潮汐测试。
        返回: client_states, available_migs, current_bandwidth
        """
        self.current_epoch += 1
        return self._step_tide()
        
    def _step_tide(self):
        # ---------------------------------------------------------
        # 1. 确定本轮宏观潮汐参数
        # ---------------------------------------------------------
        if self.current_epoch <= 50:
            # 阶段一 (稳态)
            self.current_N = np.random.randint(8, 12)
            self.current_migs = 2
            self.current_bandwidth = self.base_bandwidth
            
        elif 51 <= self.current_epoch <= 100:
            # 阶段二 (突发涌入 + 算力红利)
            self.current_N = np.random.randint(15, 20) 
            self.current_migs = 5
            self.current_bandwidth = self.base_bandwidth
            
        else:
            # 阶段三 (恢复常态 + 突发通信拥塞)
            self.current_N = np.random.randint(8, 12)
            self.current_migs = 2
            self.current_bandwidth = self.base_bandwidth * 0.2
            
        # 安全断言：活跃车辆不能超过注册上限
        self.current_N = min(self.current_N, self.max_vehicles)

        # ---------------------------------------------------------
        # 2. 模拟车辆移动性 (动态抽取本轮在场车辆 ID)
        # ---------------------------------------------------------
        # 从 0 ~ max_vehicles-1 的全局 ID 中无放回地随机抽取 N 辆车
        # 完美模拟车辆驶入驶出的“动态参与”现象
        self.active_vehicle_ids = np.random.choice(
            self.max_vehicles, size=self.current_N, replace=False
        )
        
        # 3. 组装并返回状态
        client_states = self._generate_state_for_active_vehicles()
        return client_states, self.current_migs, self.current_bandwidth, self.active_vehicle_ids
        
    def _generate_state_for_active_vehicles(self):
        """
        根据抽签选出的活跃车辆 ID，从注册表中提取固化属性，并附加快衰落信道
        """
        N = self.current_N
        
        # 1. 无线信道状态 h_n (高频变化，每轮刷新合理)
        h_n = np.random.rayleigh(scale=1.0, size=(N, 1))
        
        # 2. 提取这批在场车辆的固化算力 f_n
        f_n = self.vehicle_pool_computes[self.active_vehicle_ids].reshape(N, 1)
        
        # 3. 提取这批在场车辆的真实标签分布 v_n
        v_n = self.vehicle_pool_labels[self.active_vehicle_ids]
        
        # 组合状态空间 (N, 12)
        client_states = np.concatenate([h_n, f_n, v_n], axis=1)
        return client_states
        
    def calc_wireless_transmission_delay(self, cluster_choices, bw_weights, smashed_data_sizes_bytes, channel_gains):
        """
        根据动作和真实产生的张量大小，计算木桶传输时延。
        
        输入:
        - cluster_choices: (N,) 设备选择的簇 ID
        - bw_weights: (N,) 带宽分配权重
        - smashed_data_sizes_bytes: (N,) 由 PyTorch 真实测出的 Smashed Data 字节数
        
        返回:
        - max_tx_delay_per_cluster: (K,) 各 MIG 簇接收特征所需的最长传输时间 (木桶短板)
        """
        N = len(cluster_choices)
        h_n = np.asarray(channel_gains, dtype=np.float64)
        if h_n.shape != (N,) or not np.isfinite(h_n).all() or (h_n < 0.0).any():
            raise ValueError("channel_gains must be a finite non-negative array of shape (N,)")
        tx_delays = np.zeros(N)
        max_tx_delay_per_cluster = np.zeros(self.max_migs)

        for k in range(self.max_migs):
            mask = (cluster_choices == k)
            if np.sum(mask) == 0:
                continue
                
            # MIG 簇均分总带宽，簇内根据 \beta_n 进行 Softmax 加权分配
            cluster_band = self.current_bandwidth / self.current_migs
            w = bw_weights[mask]
            # 稳定的 softmax 避免溢出
            exp_w = np.exp(w - np.max(w))
            alloc_ratios = exp_w / np.sum(exp_w)
            allocated_bw = alloc_ratios * cluster_band
            
            # 香农定理计算上行速率
            # SNR 正比于信道增益 h_n，假设基础发送信噪比为 10dB (即倍数 10)
            snr = 10.0 * h_n[mask]
            rates_bps = allocated_bw * np.log2(1 + snr)
            
            # 计算传输时间：(真实字节数 * 8) / 传输速率
            delays = (smashed_data_sizes_bytes[mask] * 8) / (rates_bps + 1e-9)
            tx_delays[mask] = delays
            
            # 簇内同步接收必须等待最慢的那个到达 (木桶效应)
            max_tx_delay_per_cluster[k] = np.max(delays)
            
        return max_tx_delay_per_cluster