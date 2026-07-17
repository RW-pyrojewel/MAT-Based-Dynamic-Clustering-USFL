import csv
import json
import os
from datetime import datetime

class SimulationLogger:
    """
    6G AI-RAN 仿真数据日志记录器
    用于记录 150 轮潮汐演进中各个基线算法的性能指标，并导出为标准格式供后续论文绘图使用。
    """
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # 存储所有算法的仿真记录
        # 结构: { "算法名称": [ { 记录字典 }, ... ] }
        self.records = {}

    def log_step(self, algo_name, epoch, n_clients, n_migs, bandwidth, total_delay, tx_delay=0.0, comp_delay=0.0):
        """
        记录单个算法在当前 Epoch 的所有关键物理状态与性能反馈。
        
        参数:
        - algo_name: 算法名称 (例如 'MAT-RL', 'Adapted-CPSL')
        - epoch: 当前通信轮次 (1~150)
        - n_clients: 当前时隙活跃车辆数 N(t)
        - n_migs: 当前可用虚拟簇 (MIG) 切片数
        - bandwidth: 当前系统总带宽 (Hz)
        - total_delay: 系统单轮总时延，即木桶效应后最慢簇的时延 (秒)
        - tx_delay: (可选) 最大传输时延分量 (秒)
        - comp_delay: (可选) 最大计算时延分量 (秒)
        """
        if algo_name not in self.records:
            self.records[algo_name] = []
            
        record = {
            "epoch": epoch,
            "n_clients": n_clients,
            "n_migs": n_migs,
            "bandwidth_mhz": round(bandwidth / 1e6, 2),  # 转换为 MHz 方便阅读
            "total_delay_ms": round(total_delay * 1000, 4), # 转换为毫秒(ms) 方便与论文图 1-1 纵坐标对齐
            "tx_delay_ms": round(tx_delay * 1000, 4),
            "comp_delay_ms": round(comp_delay * 1000, 4)
        }

        print(f"[Logger] {algo_name} | Epoch {epoch}: N={n_clients}, MIGs={n_migs}, BW={record['bandwidth_mhz']} MHz, Total Delay={record['total_delay_ms']} ms")
        
        self.records[algo_name].append(record)

    def export_to_csv(self, filename_prefix="simulation_results"):
        """将记录导出为 CSV 文件（按算法拆分为多个文件，适合 Excel/Origin 导入画图）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files = []
        
        for algo_name, data in self.records.items():
            if not data:
                continue
                
            # 过滤掉特殊字符以作文件名
            safe_algo_name = algo_name.replace(" ", "_").replace("-", "_")
            filename = os.path.join(self.log_dir, f"{filename_prefix}_{safe_algo_name}_{timestamp}.csv")
            
            # 提取表头
            fieldnames = data[0].keys()
            
            with open(filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
                
            exported_files.append(filename)
            
        print(f"[Logger] 仿真数据已成功导出至 CSV: {self.log_dir} 目录下")
        return exported_files

    def export_to_json(self, filename="simulation_results.json"):
        """将所有算法的对比记录导出为单一 JSON 文件（适合 Python/Matplotlib 画图脚本一次性读取比对）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.log_dir, f"{filename.split('.')[0]}_{timestamp}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=4, ensure_ascii=False)
            
        print(f"[Logger] 仿真结构化数据已成功导出至 JSON: {filepath}")
        return filepath