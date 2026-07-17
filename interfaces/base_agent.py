import numpy as np

class BaseAgent:
    """
    6G AI-RAN 动态分簇 USFL 仿真环境: 统一的基线代理接口

    所有算法 (MAT-RL 以及三⼤ Baseline) 必须继承此抽象类并实现 step 方法。
    """
    def __init__(self, agent_name: str = "BaseAgent"):
        self.agent_name = agent_name

    def step(self, active_clients_state: np.ndarray, available_migs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        环境在每个通信轮次 (Epoch/Step) 调⽤此⽅法以获取智能体的动作决策。
        
        Args:
            active_clients_state (np.ndarray): 
                shape 为 (N, state_dim) 的 2D Numpy 数组，代表当前时隙活跃的 N 个设备的异构状态。
                N 的⼤⼩是动态变⻓的 (动态参与场景)。
                state_dim 包含各设备的物理特征 (如信道增益、电池余量、本地数据量、算⼒等)。
            
            available_migs (int):
                当前时隙 AI-RAN 基站实际空闲可⽤的 MIG 算⼒切⽚数量 (即物联⽹可⽀撑的最⼤并发虚拟簇数量)。
                
        Returns:
            tuple 包含 4 个 1D Numpy 数组，⻓度均为 N，分别对应每个活跃设备的决策：
            
            - client_cluster_choices (np.ndarray): 
                (N,) 整数数组。每个设备选择加⼊的虚拟簇 ID。
                有效值范围建议为 [0, available_migs - 1]。若指定超出该范围的值，环境将实施严厉的排队惩罚。
                
            - client_l1_choices (np.ndarray): 
                (N,) 整数数组。每个设备的 Part A 前端切分点层数。
                
            - client_l2_choices (np.ndarray): 
                (N,) 整数数组。每个设备的 Part C 后端切分点层数。
                要求 l1 <= l2。
                
            - client_bw_weights (np.ndarray): 
                (N,) 浮点数组。每个设备对所在虚拟簇内⽆线带宽资源的连续申请权重。
                建议取值在 (0, 1] 之间，底层环境会对同⼀虚拟簇内的设备权重执⾏ Softmax 归一化进⾏实际分配。
        """
        raise NotImplementedError("Subclasses must implement the step() method.")

    def update_policy(self, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray, **kwargs):
        """
        [可选实现] ⽤于基于强化学习的算法 (如 MAT-RL, Adapted-PCSFL) 在环境反馈后更新其内部⽹络权重。
        启发式或静态算法 (如 Adapted-CPSL, Adapted-ClusterSFL) 可选择不实现此⽅法。
        
        Args:
            rewards: 当前时隙获得的奖励。
            next_states: 下⼀时隙的状态。
            dones: 标记序列是否结束。
            **kwargs: 其他算法特定的更新参数 (如 action_log_probs, values 等)。
        """
        pass

    def __str__(self):
        return f"Agent: {self.agent_name}"