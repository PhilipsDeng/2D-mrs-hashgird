import torch
import tinycudann as tcnn

# 定义输入维度和编码配置
n_input_dims = 2
encoding_config = {
    "otype": "HashGrid",
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "base_resolution": 16,
    "max_resolution": 512
}

# 创建 Encoding 实例
hashgrid = tcnn.Encoding(n_input_dims, encoding_config, dtype=torch.float32).cuda()

# 创建一个随机输入张量
input = torch.randn(10, n_input_dims).cuda()
print("input:", input.shape)

# in:([10, 3])  -->  out:([10, 32])

# 对输入张量进行编码
encoded_tensor = hashgrid(input)

# 输出编码后的张量的形状
print(encoded_tensor.shape)


coordinates = torch.tensor([[0.1, 0.2]], dtype=torch.float32).cuda()

# 获取特征向量
features = hashgrid(coordinates)

print(features.shape)  # 输出特征向量的形状
print(features)  # 输出特征向量