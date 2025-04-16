import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads # 注意力头的数量
        self.d_model = d_model # 词向量维度
        self.d_k = d_model // num_heads # 每个头的维度（整除）

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Step 1: 线性投影 + 分头
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V shape: [batch_size, num_heads, seq_len, d_k]

        # Step 2: 缩放点注意力
        scores = torch.matmul(Q, K.transpose(-2, -1) / (self.d_k ** 0.5))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V); # [batch, heads, seq_len, d_k]

        # Step 3: 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Step 4: 最后线性层
        output = self.W_o(attn_output)

        return output

# 测试 case
batch_size = 2
seq_len = 5
d_model = 12
num_heads = 3

x = torch.randn(batch_size, seq_len, d_model)
mha = MultiHeadAttention(d_model, num_heads)

output = mha(x)

print(x.shape)
print(output)