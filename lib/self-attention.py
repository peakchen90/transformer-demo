import torch
import torch.nn.functional as F
from softmax import *

################## 公式 ###########################
# Attention(Q, K, V) = softmax(Q × Kᵀ / √dₖ) × V
###################################################

# 1. 输入：3 个词，每个词是 4 维向量
x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 1.0],
])

# 2. （模拟）初始化线性变换参数（W_q, W_k, W_v）。真实场景这些数据应该是被训练出来的。
d_model = x.shape[1]  # 每个词的维度
W_q = torch.randn(d_model, d_model)
W_k = torch.randn(d_model, d_model)
W_v = torch.randn(d_model, d_model)

# 3. 计算 Q, K, V
Q = x @ W_q
K = x @ W_k
V = x @ W_v

# 4. 计算注意力分数（缩放内积注意力）
dk = Q.shape[-1] # 这表示 Q 的特征维度（即 d_k）
scores = (Q @ K.T) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

# 5. softmax 归一化
weights = F.softmax(scores, dim=-1)

# 6. 计算输出：加权 V
output = weights @ V

print(output)

# 封装成函数
def scaled_dot_product_attention(Q, K, V):
    # 4. 计算注意力分数（缩放内积注意力）
    dk = Q.shape[-1] # 这表示 Q 的特征维度（即 d_k）
    scores = (Q @ K.T) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

    # 5. softmax 归一化
    weights = F.softmax(scores, dim=-1)

    # 6. 计算输出：加权 V
    output = weights @ V

    return output, weights