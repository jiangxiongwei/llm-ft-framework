# 导入相关需要的包
import math
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings(action="ignore")


class SelfAttV1(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttV1, self).__init__()
        self.hidden_dim = hidden_dim
        # 一般 Linear 都是默认有 bias
        # 一般来说， input dim 的 hidden dim
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        # X shape is: (batch, seq_len, hidden_dim)， 一般是和 hidden_dim 相同
        # 但是 X 的 final dim 可以和 hidden_dim 不同
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)

        # shape is: (batch, seq_len, seq_len)
        # torch.matmul 可以改成 Q @ K.transpose(-1, -2)
        # 其中 K 需要改成 shape 为： (batch, hidden_dim, seq_len)
        attention_value = torch.matmul(Q, K.transpose(-1, -2))
        attention_wight = torch.softmax(
            attention_value / math.sqrt(self.hidden_dim), dim=-1
        )
        # print(attention_wight)
        # shape is: (batch, seq_len, hidden_dim)
        output = torch.matmul(attention_wight, V)
        return output

# 随机三维张量
X = torch.rand(3, 2, 4)
print(X)
net = SelfAttV1(4)
net(X)

# 保存模型
model_path = 'self_att_v1_model.pth'
torch.save(net.state_dict(), model_path)
print(f"模型已保存到 {model_path}")

# 加载模型
loaded_net = SelfAttV1(4)
loaded_net.load_state_dict(torch.load(model_path))
loaded_net.eval()  # 设置为评估模式

# 使用加载的模型进行推理
loaded_output = loaded_net(X)
print("使用加载的模型进行推理的输出:", loaded_output)