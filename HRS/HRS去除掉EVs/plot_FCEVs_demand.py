import matplotlib.pyplot as plt
import numpy as np
from components import FCEVDemandGenerator

def main():
    # 实例化你项目中的需求生成器
    generator = FCEVDemandGenerator()

    hdv_counts = []
    ldv_counts = []

    # 模拟一天 96 个时间步 (15分钟一步)
    for step in range(96):
        # 调用 generate_vehicles 方法生成当前时间步的车辆对象列表
        vehicles = generator.generate_vehicles(step)
        
        # 统计 350-bar 和 700-bar 的车辆数
        # 根据你 components.py 的结构，vehicles 列表中是具体的车辆对象，包含 target_pressure 属性
        hdv = sum(1 for v in vehicles if getattr(v, 'target_pressure', 0) == 350)
        ldv = sum(1 for v in vehicles if getattr(v, 'target_pressure', 0) == 700)
        
        hdv_counts.append(hdv)
        ldv_counts.append(ldv)

    # x轴表示 0 到 95 个时间步
    x_axis = np.arange(96)

    # 全局绘图设置
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(8, 4.5))

    # 绘制曲线图
    plt.plot(x_axis, hdv_counts, label='350-bar HDVs', color='#1f77b4', linewidth=2)
    plt.plot(x_axis, ldv_counts, label='700-bar LDVs', color='#ff7f0e', linewidth=2)
    plt.plot(x_axis, np.array(hdv_counts) + np.array(ldv_counts), label='Total', color='#2ca02c', linewidth=1.5, linestyle='--', alpha=0.8)

    # 设置图表元素
    plt.title("Simulated FCEV Arrivals (24 Hours)")
    plt.xlabel("Time Step (15 min)")
    plt.ylabel("Number of Arrivals")
    
    # 设置 x 轴刻度，使其更具可读性 (每 16 步为 4 小时)
    xticks = np.arange(0, 97, 16)
    xticklabels = [f"{int(x/4):02d}:00" for x in xticks]
    plt.xticks(xticks, xticklabels)
    
    plt.xlim(0, 95)
    
    # 添加网格线以增强可读性
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')

    # 紧凑布局并保存高清图片
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()