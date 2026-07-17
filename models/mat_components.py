import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HeterogeneousEncoder(nn.Module):
    """
    处理异构设备状态的 Transformer 编码器。
    天然支持变长序列输入，适应动态参与。
    """
    def __init__(self, state_dim, hidden_dim, num_heads=4, num_layers=2):
        super(HeterogeneousEncoder, self).__init__()
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, client_states):
        """
        client_states: Shape (Batch, N, state_dim)
        返回全局融合后的节点特征表示。
        """
        # (Batch, N, hidden_dim)
        x = F.relu(self.state_embed(client_states))
        # 通过自注意力机制交互，隐式聚拢标签分布互补的节点 (降低 KL 散度)
        # 和算力/信道互补的节点。
        encoded_states = self.transformer(x)
        return encoded_states

class AutoregressiveDecoder(nn.Module):
    """
    基于多智能体优势分解定理的因果自回归解码器。
    顺序生成设备的动作，并用掩码防止并发资源碰撞。
    """
    def __init__(self, hidden_dim, num_migs, num_cut_layers, num_heads=4):
        super(AutoregressiveDecoder, self).__init__()
        self.num_migs = num_migs
        self.num_cut_layers = num_cut_layers
        self.hidden_dim = hidden_dim
        
        # 为了进行自回归，我们需要知道“已经做出的决策”对当前资源的影响。
        # 我们用一个简单的线性层来编码当前累积的集群分配状态。
        # 输入维度: (num_migs, 2) -> 分别表示每个 MIG 已分配的设备数和累计占用带宽。
        self.context_embed = nn.Linear(num_migs * 2, hidden_dim)
        
        # 解码器层 (只用一层进行简单融合，实际中可加深)
        # 这里为了简化，我们使用 MultiheadAttention 直接融合当前节点特征和历史决策上下文
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # 设备级动作头
        self.cluster_head = nn.Linear(hidden_dim, num_migs)
        self.bw_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # 输出 0-1 的权重
        )

    def forward(self, encoded_states, available_migs):
        """
        自回归生成设备层动作。
        encoded_states: (B, N, hidden_dim) 来自 Encoder 的特征。
        available_migs: 当前空闲的 MIG 数量 (为了简化，假设 batch 内一致)。
        返回: (B, N) 簇选择, (B, N) 带宽权重
        """
        batch_size, N, _ = encoded_states.shape
        device = encoded_states.device
        
        cluster_choices = torch.zeros((batch_size, N), dtype=torch.long, device=device)
        bw_weights = torch.zeros((batch_size, N), dtype=torch.float32, device=device)
        
        # 维护一个历史决策的上下文状态 (每个 MIG 的分配设备数和预估占用带宽)
        # 初始状态全为 0。Shape: (B, num_migs * 2)
        history_ctx = torch.zeros((batch_size, self.num_migs * 2), dtype=torch.float32, device=device)
        
        # 顺序(自回归)为每个设备生成决策
        for n in range(N):
            # 获取当前设备的编码特征 (B, 1, hidden_dim)
            current_state = encoded_states[:, n, :].unsqueeze(1)
            
            # 编码历史分配上下文 (B, 1, hidden_dim)
            ctx_embedded = F.relu(self.context_embed(history_ctx)).unsqueeze(1)
            
            # 融合当前状态与历史上下文
            # 使用 Attention 可以让当前设备根据历史状态决定最有利的资源去向
            fused_state, _ = self.attention(current_state, ctx_embedded, ctx_embedded)
            fused_state = fused_state.squeeze(1) # (B, hidden_dim)
            
            # 1. 生成簇决策
            cluster_logits = self.cluster_head(fused_state) # (B, num_migs)
            # 应用掩码：只能选择 0 到 available_migs-1 的簇
            mask = torch.full_like(cluster_logits, -1e9)
            mask[:, :available_migs] = 0
            cluster_logits = cluster_logits + mask
            
            probs = F.softmax(cluster_logits, dim=-1)
            # 训练时可以采样(Categorical)，部署时取 argmax。这里简化为取 argmax
            choice = torch.argmax(probs, dim=-1) 
            cluster_choices[:, n] = choice
            
            # 2. 生成带宽权重
            bw_w = self.bw_head(fused_state).squeeze(-1) # (B,)
            bw_weights[:, n] = bw_w
            
            # 3. 更新历史上下文 (供下一个设备参考，实现“前车之鉴”)
            # 这里简化更新逻辑：为选中的 MIG 增加 1 个设备计数，并累加部分带宽权重
            for b in range(batch_size):
                mig_idx = choice[b].item()
                history_ctx[b, mig_idx * 2] += 1.0 # 增加设备计数
                history_ctx[b, mig_idx * 2 + 1] += bw_w[b].item() # 增加带宽权重估计
                
        return cluster_choices, bw_weights