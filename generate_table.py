import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# 从script.py运行结果中提取的数据
gpu_memory_usage = [
    {"ft_k": 8, "packet_pool_size": 1000000, "output_file": "flow_statics_8.csv", 
     "average_flow_size": 10000, "flow_time_range": 10000, "max_gpu_memory_mb": 909, "max_gpu_memory_gb": 0.89},
    {"ft_k": 16, "packet_pool_size": 5000000, "output_file": "flow_statics_16.csv", 
     "average_flow_size": 10000, "flow_time_range": 10000, "max_gpu_memory_mb": 2141, "max_gpu_memory_gb": 2.09},
    {"ft_k": 32, "packet_pool_size": 10000000, "output_file": "flow_statics_32.csv", 
     "average_flow_size": 10000, "flow_time_range": 10000, "max_gpu_memory_mb": 4385, "max_gpu_memory_gb": 4.28},
    {"ft_k": 48, "packet_pool_size": 20000000, "output_file": "flow_statics_48.csv", 
     "average_flow_size": 10000, "flow_time_range": 10000000, "max_gpu_memory_mb": 9449, "max_gpu_memory_gb": 9.23},
    {"ft_k": 64, "packet_pool_size": 50000000, "output_file": "flow_statics_64.csv", 
     "average_flow_size": 10000, "flow_time_range": 20000000, "max_gpu_memory_mb": 22329, "max_gpu_memory_gb": 21.81}
]

simulation_duration = [5.44302, 8.72547, 22.98910, 74.39500, 222.73400]

# 创建数据框
df = pd.DataFrame(gpu_memory_usage)
df['simulation_duration'] = simulation_duration

# 格式化数据
df_formatted = df[[
    'ft_k', 'packet_pool_size', 'average_flow_size', 'flow_time_range',
    'max_gpu_memory_gb', 'max_gpu_memory_mb', 'simulation_duration'
]].copy()

df_formatted.columns = [
    'FatTree k', 'Packet Pool Size', 'Average Flow Size', 'Flow Time Range (ns)',
    'Max GPU Memory (GB)', 'Max GPU Memory (MB)', 'Simulation Duration (s)'
]

# 格式化数值显示
df_formatted['Packet Pool Size'] = df_formatted['Packet Pool Size'].apply(lambda x: f"{x:,}")
df_formatted['Average Flow Size'] = df_formatted['Average Flow Size'].apply(lambda x: f"{x:,}")
df_formatted['Flow Time Range (ns)'] = df_formatted['Flow Time Range (ns)'].apply(lambda x: f"{x:,}")
df_formatted['Max GPU Memory (GB)'] = df_formatted['Max GPU Memory (GB)'].apply(lambda x: f"{x:.2f}")
df_formatted['Max GPU Memory (MB)'] = df_formatted['Max GPU Memory (MB)'].apply(lambda x: f"{x:,}")
df_formatted['Simulation Duration (s)'] = df_formatted['Simulation Duration (s)'].apply(lambda x: f"{x:.5f}")

print("整理后的表格数据:")
print(df_formatted.to_string(index=False))

# 创建PDF文件
with PdfPages('/home/mobinets/GeDES_artifact/simulation_results.pdf') as pdf:
    # 创建表格页面
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(
        cellText=df_formatted.values,
        colLabels=df_formatted.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # 设置标题
    plt.title('GeDES Simulation Results Summary', fontsize=16, pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # 创建统计信息页面
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # 计算统计信息
    stats_text = f"""
    Simulation Statistics Summary
    =============================
    
    Total Simulations: {len(df)}
    
    Memory Usage Range: {df['max_gpu_memory_gb'].min():.2f} - {df['max_gpu_memory_gb'].max():.2f} GB
    Average Memory Usage: {df['max_gpu_memory_gb'].mean():.2f} GB
    
    Duration Range: {df['simulation_duration'].min():.2f} - {df['simulation_duration'].max():.2f} seconds
    Average Duration: {df['simulation_duration'].mean():.2f} seconds
    Total Simulation Time: {df['simulation_duration'].sum():.2f} seconds
    
    FatTree Configurations Tested: k={', '.join(map(str, sorted(df['ft_k'].unique())))}
    """
    
    ax.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top', transform=ax.transAxes)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print("\nPDF文件已保存到: /home/mobinets/GeDES_artifact/simulation_results.pdf")