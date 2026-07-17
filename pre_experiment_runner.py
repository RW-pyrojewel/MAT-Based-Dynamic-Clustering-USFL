import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体，防止图表乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 获取文件路径
root_path = os.path.dirname(os.path.abspath(__file__)) + os.sep

# ==========================================
# 1. 物理环境与基础参数设定
# ==========================================
T_local_comp = 30               
T_unit_edge_comp = 20           
T_base_comm = 50                
collision_factor = 0.1          
epochs = 150                    

# 加入多组静态策略，形成完整的 Benchmark 矩阵
k_strategies = {
    '策略 A (静态大簇 K=1)': 1,                  
    '策略 B (静态小簇 K=4)': 4,                  
    '策略 C (启发式自适应 / 如 ClusterSFL)': 'heuristic', 
    '策略 D (传统 DRL 并发决策 / 如 PCSFL)': 'traditional_drl' 
}
results = {name: [] for name in k_strategies}

# --- 内部状态初始化 ---
D_K_current = 2
D_EMA_BW = 1.0       
alpha_ema = 0.8
drl_eps = 0.0 # 传统 DRL 探索率

# ==========================================
# 2. 真实物理引擎与主仿真循环
# ==========================================
for epoch in range(epochs):
    # --- 物理环境潮汐演进 (加入动态 N) ---
    if epoch < 50:
        N_devices = 10; MIG_available = 2; BW_scale = 1.0 # 稳态
    elif epoch < 100:
        N_devices = 18; MIG_available = 5; BW_scale = 1.0 # 车辆突发涌入 + 算力红利释放
    else:
        N_devices = 10; MIG_available = 2; BW_scale = 0.2 # 车辆恢复常规 + 突发通信拥塞
        
    # 传统 DRL 智能体触发重新探索
    if epoch == 50 or epoch == 100:
        drl_eps = 0.85
    drl_eps = max(0.05, drl_eps * 0.90) 

    for name, K_val in k_strategies.items():
        C_max = 0 
        
        # ========================================
        # 智能体决策阶段
        # ========================================
        if name == '策略 C (启发式自适应 / 如 ClusterSFL)':
            D_EMA_BW = alpha_ema * D_EMA_BW + (1 - alpha_ema) * BW_scale
            assumed_MIG = 2 # 算力致盲
            
            best_cost = float('inf')
            next_k = D_K_current
            for k_cand in [max(1, D_K_current - 1), D_K_current, min(N_devices, D_K_current + 1)]:
                est_comp = np.ceil(k_cand / assumed_MIG) * np.ceil(N_devices / k_cand) * T_unit_edge_comp
                est_comm = (T_base_comm / D_EMA_BW) * (1 + collision_factor * (k_cand - 1) / D_EMA_BW)
                if (est_comp + est_comm) < best_cost:
                    best_cost = est_comp + est_comm
                    next_k = k_cand
            K = next_k; D_K_current = next_k
            C_max = np.ceil(N_devices / K) 
            
        elif name == '策略 D (传统 DRL 并发决策 / 如 PCSFL)':
            if epoch >= 50 and epoch < 100:
                # 阶段二：动态参与 (N=18)。Zero-padding 扩维导致策略退化
                if np.random.rand() < drl_eps:
                    K = np.random.randint(2, 6)
                    assignments = np.random.randint(0, K, N_devices)
                    C_max = np.max(np.bincount(assignments, minlength=K))
                else:
                    K = 3; C_max = np.ceil(N_devices / K) + 2 
                    
            elif epoch >= 100:
                # 阶段三：突发拥塞。并发动作引发严重碰撞
                if np.random.rand() < drl_eps:
                    K = np.random.randint(1, 4)
                    assignments = np.random.randint(0, K, N_devices)
                    C_max = np.max(np.bincount(assignments, minlength=K))
                else:
                    K = 2; C_max = np.ceil(N_devices / K)
            else:
                K = 2; C_max = np.ceil(N_devices / K)
                
        else: # 静态策略 A 和 B
            K = K_val
            C_max = np.ceil(N_devices / K)
        
        # ========================================
        # 真实物理延迟结算阶段
        # ========================================
        real_waves = np.ceil(K / MIG_available)
        T_comp = real_waves * C_max * T_unit_edge_comp
        overhead_penalty = 1 + collision_factor * (K - 1) / BW_scale
        T_comm = (T_base_comm / BW_scale) * overhead_penalty
        
        T_total = T_local_comp + T_comp + T_comm + np.random.normal(0, 3)
        results[name].append(T_total)

# ==========================================
# 3. 结果可视化绘制
# ==========================================
plt.figure(figsize=(12, 6.5))
# 配色优化：大簇深蓝，小簇青色，启发式绿色，DRL橙色
colors = ['#2980b9', '#1abc9c', '#27ae60', '#e67e22'] 
linestyles = ['-', '-', '-.', ':']

for idx, (name, delay_list) in enumerate(results.items()):
    lw = 2.5 if idx >= 2 else 2.0
    alpha = 0.9 if idx >= 2 else 0.75
    plt.plot(range(epochs), delay_list, label=name, color=colors[idx], 
             linewidth=lw, linestyle=linestyles[idx], alpha=alpha)

plt.axvspan(0, 50, color='gray', alpha=0.1, label='阶段一: 稳态 (N=10, MIG=2, BW=1.0)')
plt.axvspan(50, 100, color='#3498db', alpha=0.08, label='阶段二: 车辆涌入+算力红利 (N=18, MIG=5)')
plt.axvspan(100, 150, color='#e74c3c', alpha=0.08, label='阶段三: 通信拥塞 (N=10, MIG=2, BW=0.2)')

plt.title('6G VEC 环境下引入“动态参与与资源潮汐”的单轮 SFL 时延性能对比', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('全局训练轮次 (Global Epochs)', fontsize=14)
plt.ylabel('单轮总时延 $T_{epoch}$ (ms)', fontsize=14)
plt.xlim(0, 150); plt.ylim(100, 1100)

plt.text(75, 900, '【问题三】传统DRL\n遭遇车辆涌入(N变长),补零扩维\n导致策略坍塌与极端时延', 
         ha='center', va='center', fontsize=11, color='#e67e22', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e67e22', pad=3))
plt.text(75, 450, '【问题一&二】静态大簇无法分裂\n启发式“算力致盲”未利用红利\n均被涌入的车辆拖垮', 
         ha='center', va='center', fontsize=11, color='#2c3e50', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=3))
plt.text(125, 780, '【问题一&三】静态小簇产生信令风暴\n传统DRL缺乏因果掩码引发并发碰撞\n均产生剧烈的排队与震荡', 
         ha='center', va='center', fontsize=11, color='#c0392b', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e67e22', pad=3))

plt.legend(loc='upper left', fontsize=10, framealpha=0.95)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(root_path, 'figs', 'pre_experiment.png'), dpi=300, bbox_inches='tight')
print("仿真完成！图表已保存至 'pre_experiment.png'")
plt.show()