import torch
import torch.nn as nn
import numpy as np

# 假设我们在同级目录下运行，引入之前定义的 BaseAgent 接口和 MAT 组件
from interfaces.base_agent import BaseAgent
from models.mat_components import HeterogeneousEncoder, AutoregressiveDecoder
import torch.nn.functional as F

class MATAgent(BaseAgent):
    """
    基于多智能体 Transformer 的 6G AI-RAN 动态分簇智能体
    实现了严格的“分层决策”逻辑：
    1. 设备级 (Device-level)：自回归生成加入的虚拟簇 (cluster_choices) 和连续带宽权重 (bw_weights)。
    2. 簇级 (Cluster-level)：对映射到同一簇的设备特征进行聚合，集中预测该簇统一的前后端切分点 (l1, l2)。
    """
    def __init__(self, state_dim, hidden_dim=128, num_migs=7, num_cut_layers=7, device='cpu'):
        super().__init__(agent_name="MAT-RL Agent (Proposed)")
        self.device = device
        self.num_migs = num_migs
        self.num_cut_layers = num_cut_layers
        
        # ---------------------------------------------------------
        # 1. 设备级网络：处理异构状态并自回归分配物理资源
        # ---------------------------------------------------------
        self.encoder = HeterogeneousEncoder(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
        self.decoder = AutoregressiveDecoder(hidden_dim=hidden_dim, num_migs=num_migs, num_cut_layers=num_cut_layers).to(device)
        
        # ---------------------------------------------------------
        # 2. 簇级决策头：决定统一的物理切分点 (l1, l2)
        # ---------------------------------------------------------
        # 为了预测 l1 和 l2，且保证 l1 < l2，我们分别使用两个线性头
        self.cluster_l1_head = nn.Linear(hidden_dim, num_cut_layers).to(device)
        self.cluster_l2_head = nn.Linear(hidden_dim, num_cut_layers).to(device)
        
        # ---------------------------------------------------------
        # 3. Critic 网络价值头 (复用 Encoder，极简设计)
        # ---------------------------------------------------------
        # [修改] 遵循 MAT 核心思想，摒弃独立的 critic_encoder
        # 直接使用共享的 Encoder 提取的特征来评估全局价值
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        ).to(device)
        
        # ---------------------------------------------------------
        # 4. 优化器配置 (联合更新)
        # ---------------------------------------------------------
        # [修改] Actor 和 Critic 的损失将联合驱动底层 Encoder 更新
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': 3e-4},       # 共享底层
            {'params': self.decoder.parameters(), 'lr': 3e-4},
            {'params': self.cluster_l1_head.parameters(), 'lr': 3e-4},
            {'params': self.cluster_l2_head.parameters(), 'lr': 3e-4},
            {'params': self.value_head.parameters(), 'lr': 1e-3}     # 价值头
        ])
        
        # PPO 超参数
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01

    def step(self, active_clients_state: np.ndarray, available_migs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        覆盖 BaseAgent 的核心决策接口。
        将 numpy 输入转换为 Tensor，经过 MAT 网络，再返回 numpy 格式的 4 个一维数组。
        """
        # 将 numpy 数组转为 (Batch, N, state_dim) 格式的 Tensor
        # 在线上部署/推断时，Batch 维度恒为 1
        states_tensor = torch.tensor(active_clients_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 为了反向传播（训练阶段），这里可以使用 torch.no_grad() 如果只是纯 inference。
        # 考虑到这里既用于仿真推演，我们保持计算图连通（除非在 rollout 时手动包裹 with torch.no_grad():）
        
        # =========================================================
        # 阶段一：设备级动作生成 (自回归规避碰撞)
        # =========================================================
        encoded_states = self.encoder(states_tensor) # (1, N, hidden_dim)
        
        # 返回形状均为 (1, N) 的 Tensor
        cluster_choices_t, bw_weights_t = self.decoder(encoded_states, available_migs)
        
        # =========================================================
        # 阶段二：簇级切分点决策 (保证物理张量拼接维度对齐)
        # =========================================================
        N = active_clients_state.shape[0]
        l1_choices_t = torch.zeros((1, N), dtype=torch.long, device=self.device)
        l2_choices_t = torch.zeros((1, N), dtype=torch.long, device=self.device)
        
        # 遍历当前可用的每一个 MIG 虚拟簇
        for k in range(available_migs):
            # 找到哪些设备选择了加入簇 k
            mask = (cluster_choices_t == k).unsqueeze(-1).float() # (1, N, 1)
            
            # 如果当前簇没有设备加入，则跳过
            if mask.sum() == 0:
                continue
                
            # 1. 特征聚合 (Mean Pooling)：将簇 k 内所有设备的隐藏特征进行聚合
            sum_feats = (encoded_states * mask).sum(dim=1) # (1, hidden_dim)
            count = mask.sum(dim=1) + 1e-8
            cluster_ctx = sum_feats / count # 得到代表该虚拟簇整体特征的上下文向量
            
            # 2. 预测前端切分点 l1
            l1_logits = self.cluster_l1_head(cluster_ctx) # (1, num_cut_layers)
            l1_probs = torch.softmax(l1_logits, dim=-1)
            l1 = torch.argmax(l1_probs, dim=-1) # (1,)
            
            # 3. 预测后端切分点 l2，且强制施加掩码保证 l1 < l2
            l2_logits = self.cluster_l2_head(cluster_ctx) # (1, num_cut_layers)
            
            # 构造掩码：不允许 l2 <= l1
            l2_mask = torch.full_like(l2_logits, -1e9)
            l1_val = l1.item()
            if l1_val + 1 < self.num_cut_layers:
                l2_mask[:, l1_val + 1:] = 0
            else:
                # 极端边界情况保护：如果 l1 已经选到了倒数第一层，l2 只能选最后一层
                l2_mask[:, -1] = 0 
                
            l2_logits = l2_logits + l2_mask
            l2_probs = torch.softmax(l2_logits, dim=-1)
            l2 = torch.argmax(l2_probs, dim=-1) # (1,)
            
            # 4. 决策广播：将算出的簇级 (l1, l2) 强行赋给隶属于该簇的所有设备
            # 从而完美避开物理层的 Tensor 维度冲突
            l1_choices_t = torch.where(cluster_choices_t == k, l1, l1_choices_t)
            l2_choices_t = torch.where(cluster_choices_t == k, l2, l2_choices_t)

        # =========================================================
        # 收尾：转换为 Numpy 格式返回给底层物理环境
        # =========================================================
        cluster_choices = cluster_choices_t.squeeze(0).cpu().numpy()
        bw_weights = bw_weights_t.squeeze(0).detach().cpu().numpy()
        l1_choices = l1_choices_t.squeeze(0).cpu().numpy()
        l2_choices = l2_choices_t.squeeze(0).cpu().numpy()
        
        return cluster_choices, l1_choices, l2_choices, bw_weights

    def get_value(self, state: np.ndarray) -> float:
        """获取当前状态的标量价值评估"""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            # [修改] 直接调用主 Encoder 提取特征
            encoded_state = self.encoder(state_t) # (1, N, hidden_dim)
            # 对所有设备特征进行 Mean Pooling，代表整个全局环境的状态价值
            global_feature = encoded_state.mean(dim=1)   # (1, hidden_dim)
            value = self.value_head(global_feature)      # (1, 1)
        return value.item()

    def _compute_action_log_probs(self, state_t, action_dict, available_migs):
        """
        重新计算给定状态和动作的 log_prob。
        由于我们是在 rollout 时收集数据，在 PPO 更新时需要能够重新评估旧动作的概率。
        """
        # 注意：这里的实现为了展示核心逻辑做了适度简化。
        # 真实的 PPO 训练中，需要精确重构 forward_step 中的循环掩码和因果状态。
        
        encoded_states = self.encoder(state_t)
        
        # --- 1. 设备级动作 log_prob ---
        cluster_choices_t, bw_weights_t = self.decoder(encoded_states, available_migs)
        # 此处仅作示意：将确定的模型输出与收集到的真实 action 做对比计算 log_prob
        # 真实场景中，连续动作需要假定高斯分布计算，离散动作使用 Categorical 分布
        
        # 简化返回伪 log_probs 和熵 (实际需根据分布严格计算)
        dummy_log_probs = torch.zeros((1,), device=self.device, requires_grad=True)
        dummy_entropy = torch.ones((1,), device=self.device, requires_grad=True) * 0.5
        
        return dummy_log_probs, dummy_entropy

    def update_policy(self, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray, **kwargs):
        """
        基于 PPO 算法更新 Actor 和 Critic 网络权重。
        """
        # 在真实的 RL Pipeline 中，我们会积攒一个 batch 的 trajectory
        # 这里展示对单步经验或一个 mini-batch 进行更新的骨架
        
        # 提取存储的经验 (需由外层 Runner 传入 kwargs 包含 memory_buffer)
        states = kwargs.get('states')
        actions = kwargs.get('actions') # list of dicts: {'cluster':..., 'bw':..., 'l1':..., 'l2':...}
        old_log_probs = kwargs.get('old_log_probs')
        available_migs = kwargs.get('available_migs', self.num_migs)
        
        if states is None or len(states) == 0:
            return # 数据不足时不更新
            
        # --- 计算 Return 和 Advantage (此处简化为单步 GAE 骨架) ---
        returns = []
        advantages = []
        
        for i in range(len(rewards)):
            r = rewards[i]
            v_s = self.get_value(states[i])
            v_s_next = 0.0 if dones[i] else self.get_value(next_states[i])
            
            td_target = r + self.gamma * v_s_next
            td_error = td_target - v_s
            
            returns.append(td_target)
            advantages.append(td_error)
            
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        # 标准化 Advantage
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # --- PPO Epoch 迭代更新 ---
        ppo_epochs = 4
        for _ in range(ppo_epochs):
            for i in range(len(states)):
                state_t = torch.tensor(states[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # [修改] 单次前向传播，同时服务于 Critic 和 Actor
                encoded_states = self.encoder(state_t)
                
                # 1. 计算 Critic 损失
                global_feature = encoded_states.mean(dim=1)
                value_pred = self.value_head(global_feature).squeeze(0)
                critic_loss = F.mse_loss(value_pred, torch.tensor([returns[i]], device=self.device))
                
                # 2. 计算 Actor 损失
                # 注意：_compute_action_log_probs 现在应复用 encoded_states，避免重复前向计算
                new_log_probs, entropy = self._compute_action_log_probs_from_encoded(encoded_states, actions[i], available_migs)
                
                ratio = torch.exp(new_log_probs - torch.tensor(old_log_probs[i], device=self.device))
                
                surr1 = ratio * advantages_t[i]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_t[i]
                
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                
                # 3. 联合反向传播
                # 核心思想：Critic 和 Actor 产生的梯度交汇于 Encoder，促使模型学到既能准确预测价值，又能有效区分动作的通用特征表示。
                total_loss = actor_loss + self.value_coef * critic_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=0.5) # 可选：仅限制共享层梯度
                self.optimizer.step()
                
    def _compute_action_log_probs_from_encoded(self, encoded_states, action_dict, available_migs):
        """
        [新增] 优化版本：直接接受 Encoder 提取的特征，避免在 PPO 循环中重复进行 Transformer 计算
        """
        cluster_choices_t, bw_weights_t = self.decoder(encoded_states, available_migs)
        dummy_log_probs = torch.zeros((1,), device=self.device, requires_grad=True)
        dummy_entropy = torch.ones((1,), device=self.device, requires_grad=True) * 0.5
        return dummy_log_probs, dummy_entropy