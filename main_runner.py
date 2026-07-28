import torch
import numpy as np
import time

from envs.liquid_airan_env import LiquidAIRANEnv
from models.usfl_networks import ResNet18_USFL
from models.mat_agent import MATAgent
from data import CIFAR100NonIIDProvider
from utils.logger import SimulationLogger

# --- 占位用的随机基线算法 ---
from interfaces.base_agent import BaseAgent

# --- 实际基线算法
from baselines import CPSLAgent, ClusterSFLAgent, PCSFLAgent

class RandomAgent(BaseAgent):
    """
    一个随机生成动作的基线，用于在协作者代码到位前测试主循环。
    大概率会遭遇“维度冲突”的严厉惩罚。
    """
    def step(self, active_clients_state, available_migs):
        N = active_clients_state.shape[0]
        # 随机分配簇 (0 到 available_migs-1)
        client_cluster_choices = np.random.randint(0, available_migs, size=N)
        # 随机分配 L1 和 L2，确保 L1 < L2
        l1_choices = np.random.randint(0, 4, size=N)
        l2_choices = np.random.randint(l1_choices + 1, 7, size=N)
        # 随机带宽权重
        bw_weights = np.random.uniform(0, 1, size=N)
        return client_cluster_choices, l1_choices, l2_choices, bw_weights

def simulate_epoch(env, agent, resnet, epoch):
    """单轮仿真执行流"""
    # 1. 环境时序推进，获取本轮潮汐状态
    client_states, available_migs, current_bandwidth, _ = env.step()
    N = client_states.shape[0]

    print(f"\n[Epoch {epoch}] 当前车辆数: {N}, 可用 MIGs: {available_migs}, 当前带宽: {current_bandwidth:.2f} Mbps")
    
    # 2. 算法决策
    # 记录算法推理耗时 (可选，作为参考)
    t_start = time.time()
    edge_state = np.asarray([available_migs, current_bandwidth], dtype=np.float32)
    if isinstance(agent, MATAgent):
        cluster_choices, l1_choices, l2_choices, bw_weights = agent.step(client_states, available_migs, edge_state)
    else:
        cluster_choices, l1_choices, l2_choices, bw_weights = agent.step(client_states, available_migs)
    # 对于 PyTorch 模型，如果返回值是 Tensor，转为 numpy
    if isinstance(cluster_choices, torch.Tensor):
        cluster_choices = cluster_choices.detach().cpu().numpy()
        l1_choices = l1_choices.detach().cpu().numpy()
        l2_choices = l2_choices.detach().cpu().numpy()
        bw_weights = bw_weights.detach().cpu().numpy()

    # 3. 真实物理底座检验 (Tensor Shape 裁决)
    # 伪造一批输入数据给真实 PyTorch 模型，以获取张量尺寸
    dummy_input_images = torch.randn(N, 3, 32, 32)
    smashed_data_sizes_bytes = np.zeros(N)
    
    system_max_delay = 0.0
    max_tx_delay = 0.0
    max_comp_delay = 0.0
    
    # 遍历每个活跃的 MIG 算力切片
    for k in range(available_migs):
        mask = (cluster_choices == k)
        num_in_cluster = np.sum(mask)
        if num_in_cluster == 0:
            continue
            
        # --- [物理真理法庭] 检查张量维度一致性 ---
        l1_in_cluster = l1_choices[mask]
        l2_in_cluster = l2_choices[mask]
        
        # 检查是否簇内所有设备选择了相同的 L1 和 L2
        if len(np.unique(l1_in_cluster)) == 1 and len(np.unique(l2_in_cluster)) == 1:
            # 【合法】维度一致，可以进行物理特征拼接 (Concat)
            l1 = int(l1_in_cluster[0])
            l2 = int(l2_in_cluster[0])
            
            # 设备端 (Part A) 本地前向传播模拟
            with torch.no_grad():
                smashed_tensors = resnet.forward_partA(dummy_input_images[mask], l1)
            
            # 记录真实的物理字节数用于环境计算传输延迟 (Float32 = 4 bytes)
            bytes_per_sample = smashed_tensors.numel() / num_in_cluster * 4
            smashed_data_sizes_bytes[mask] = bytes_per_sample
            
            # 【非线性加速计算】MIG 接收拼接特征进行大 Batch 计算
            # 在没有真实 GPU Profiling 时，我们用简单的非线性函数替代
            # GPU 处理大 Batch 的时间增长率远低于串行
            batch_size = num_in_cluster
            # 模拟：基础延迟 0.05秒，随着 Batch Size 略微增加
            cluster_comp_delay = 0.05 + 0.005 * batch_size 
            
        else:
            # 【崩溃】维度冲突，无法拼接！
            # 必须退化为 GPU 内逐个串行计算
            for idx in np.where(mask)[0]:
                l1 = int(l1_choices[idx])
                with torch.no_grad():
                    # 单个样本的张量大小
                    smashed = resnet.forward_partA(dummy_input_images[idx:idx+1], l1)
                smashed_data_sizes_bytes[idx] = smashed.numel() * 4
                
            # 严厉的排队串行惩罚：假设每个样本串行处理耗时 0.06 秒
            cluster_comp_delay = 0.06 * num_in_cluster
            
        # 4. 环境计算无线木桶传输延迟
        # (我们将计算所有设备的延迟，但此处只需提取当前簇 k 的最大传输时间)
        tx_delays_per_cluster = env.calc_wireless_transmission_delay(cluster_choices, bw_weights, smashed_data_sizes_bytes, client_states[:, 0])
        cluster_tx_delay = tx_delays_per_cluster[k]
        
        # 5. 计算当前簇的总时延 (通信木桶 + 计算耗时)
        cluster_total_delay = cluster_tx_delay + cluster_comp_delay
        
        # 更新全系统（跨多个物理 MIG）的最慢时延
        if cluster_total_delay > system_max_delay:
            system_max_delay = cluster_total_delay
            max_tx_delay = cluster_tx_delay
            max_comp_delay = cluster_comp_delay

    return N, available_migs, current_bandwidth, system_max_delay, max_tx_delay, max_comp_delay

def main():
    print("🚀 开始启动 6G AI-RAN SFL 150 轮潮汐仿真...")
    
    # 1. 初始化模块
    data_provider = CIFAR100NonIIDProvider(num_clients=25, alpha=0.1)
    env = LiquidAIRANEnv(data_provider)
    resnet = ResNet18_USFL(num_classes=data_provider.num_classes)
    logger = SimulationLogger(log_dir="logs")

    # 获取 GPU 可用状态
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet.to(device)
    
    # 状态维度: h_n (1), f_n (1), v_n (100) = 102
    state_dim = 2 + data_provider.num_classes
    mat_agent = MATAgent(state_dim=state_dim, hidden_dim=128, num_migs=env.max_migs, device=device)
    random_agent = RandomAgent()
    cpsl_agent = CPSLAgent()
    clustersfl_agent = ClusterSFLAgent()
    pcsfl_agent = PCSFLAgent()
    
    agents = {
        "Proposed_MAT-RL": mat_agent,
        "Baseline_Random": random_agent,
        "Baseline_CPSL": cpsl_agent,
        "Baseline_ClusterSFL": clustersfl_agent,
        "Baseline_PCSFL": pcsfl_agent,
    }
    
    # 2. 开始 150 轮总循环
    for epoch in range(1, 151):
        if epoch == 51:
            print("\n🌊 [阶段二] 潮汐变化: 车辆突发涌入 + 算力红利释放")
        elif epoch == 101:
            print("\n⚠️ [阶段三] 潮汐变化: 恢复常态车辆 + 突发通信拥塞")
            
        # 针对每个算法，在相同的本轮初始状态下推演
        for agent_name, agent in agents.items():
            # 临时重置环境内的种子或状态机，保证每个算法面对相同考题
            # 此处简化，直接让所有算法基于环境最后吐出的状态计算
            env_state_backup = (env.current_N, env.current_migs, env.current_bandwidth)
            
            # 执行仿真
            N, migs, bw, total_d, tx_d, comp_d = simulate_epoch(env, agent, resnet, epoch)
            
            # 记录数据
            logger.log_step(agent_name, epoch, N, migs, bw, total_d, tx_d, comp_d)
            
            # 恢复环境状态
            env.current_N, env.current_migs, env.current_bandwidth = env_state_backup
            
        if epoch % 10 == 0:
            print(f"[{epoch}/150] {N}辆车, {migs}个MIG. MAT总时延: {total_d:.2f}s")
            
    # 3. 导出实验结果
    print("\n✅ 150 轮仿真完成！正在导出结果...")
    logger.export_to_csv()
    logger.export_to_json()
    
if __name__ == "__main__":
    main()