import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len=5000):
        super().__init__()

        # 创建一个 [max_len, d_model] 0 矩阵
        pe = torch.zeros(max_len, d_model)

        # 生成一个 [max_len, 1] 的位置索引向量，表示第几个 token
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 生成每个维度对应的分母因子（根据 d_model）。第 3 个参数 `2` 表示步长为 2
        # @see resources/positional-encoding2.png
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数维用 sin（0,2,4,...），广播乘法
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数维用 cos（1,3,5,...）
        pe[:, 1::2] = torch.cos(position * div_term)

        # shape 变为 [1, max_len, d_model]，方便后续加到 batch 上
        pe = pe.unsqueeze(0) # [1, max_len, d_model]， 方便加到 patch 上

        # 注册为 buffer，表示它不是参数，但跟模型绑定，会保存到模型文件
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# 测试 case
d_model = 4
max_len = 10

pos_encoder = PositionalEncoding(d_model, max_len)

x = torch.zeros(2, 5, d_model)

output = pos_encoder(x)

print(x)
print(output)