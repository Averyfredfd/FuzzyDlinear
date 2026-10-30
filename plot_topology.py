import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. 替换为你刚刚跑完的 V2 实验的真实 checkpoint 路径！
# 例如: './checkpoints/ETTh1_336_96_FuzzyGNN_V2_M_..._0/checkpoint.pth'
checkpoint_path = './checkpoints/ETTh1_336_96_FuzzyGNN_V2_M_FuzzyDLinear_ETTh1_ftM_sl336_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Fuzzy_GNN_V2_Residual_0/checkpoint.pth'

if not os.path.exists(checkpoint_path):
    print(f"找不到权重文件，请检查路径: {checkpoint_path}")
    exit()

# 2. 加载模型最佳权重
state_dict = torch.load(checkpoint_path, map_location='cpu')

# 3. 提取 GNN 学习到的自适应空间邻接矩阵
# 参数名对应你写的 SpatialGNNMixer 中的 spatial_adj
spatial_adj = state_dict['spatial_mixer.spatial_adj'].numpy()

# 4. 执行与模型前向传播中一样的 Sigmoid 激活，将其映射到 0~1 的关联概率
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

adj_matrix = sigmoid(spatial_adj)

# ETT 数据集的 7 个物理变量名称
# HUFL/HULL: 高压有效/无功负荷 | MUFL/MULL: 中压 | LUFL/LULL: 低压 | OT: 变压器油温
features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# 5. 使用 Seaborn 绘制高大上的热力图
plt.figure(figsize=(10, 8))
# 设置字体和清晰度
sns.set(font_scale=1.2)
ax = sns.heatmap(adj_matrix,
                 annot=True,          # 在格子里显示具体数字
                 fmt=".3f",           # 保留3位小数
                 cmap='YlOrRd',       # 工业风配色 (黄-橙-红)
                 xticklabels=features,
                 yticklabels=features,
                 linewidths=0.5)

plt.title("Learned Spatial Topology (GNN Adjacency Matrix)", fontsize=16, pad=20)
plt.xlabel("Source Node (Impact From)", fontsize=14)
plt.ylabel("Target Node (Impact To)", fontsize=14)

# 解决上下边距被截断的 matplotlib 常见 bug
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

# 保存为高清图片
output_file = "GNN_Topology_Heatmap.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"拓扑热力图已成功保存至: {output_file}")