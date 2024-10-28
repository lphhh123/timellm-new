# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MY_Conv(nn.Module):
#     def __init__(self):
#         super(MY_Conv, self).__init__()
        
#         # 定义一系列一维卷积层，每层通道数的变化为 1 -> 8 -> 32 -> 128 -> 64 -> 32 -> 1
#         self.conv1 = nn.Conv1d(in_channels=6, out_channels=8, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(8)
        
#         self.conv2 = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(32)
        
#         self.conv3 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm1d(128)
        
#         self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm1d(64)
        
#         self.conv5 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
#         self.bn5 = nn.BatchNorm1d(32)
        
#         self.conv6 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
    
#     def forward(self, x):
#         # 转置输入维度，从 [batch_size, T, 6] -> [batch_size, 6, T]
#         x = x.permute(0, 2, 1)
        
#         # 每层卷积操作：Conv1d -> BatchNorm1d -> ReLU
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = F.relu(self.bn5(self.conv5(x)))
        
#         # 最后一层不需要激活
#         x = self.conv6(x)
        
#         # 将输出转置回 [batch_size, T, 1]
#         x = x.permute(0, 2, 1)
        
#         return x
    


# # 示例输入
# batch_size, T = 4, 10  # 假设 batch_size 为 4，T 为 10
# input_tensor = torch.randn(batch_size, T, 6)

# # 创建模型实例并前向传播
# model = MYConv()
# output_tensor = model(input_tensor)

# print(f"Input shape: {input_tensor.shape}")
# print(f"Output shape: {output_tensor.shape}")
import torch
import torch.nn as nn
import torch.nn.functional as F

class MY_DEConv(nn.Module):
    def __init__(self):
        super(MY_DEConv, self).__init__()
        
        # 定义一系列一维卷积层，通道数变化：1 -> 8 -> 32 -> 128 -> 64 -> 32 -> 6
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(32)
        
        # 最后一层输出通道数变为 6
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=6, kernel_size=3, padding=1)
    
    def forward(self, x):
        # 转置输入维度，从 [batch_size, T, 1] -> [batch_size, 1, T]
        x = x.permute(0, 2, 1)
        
        # 每层卷积操作：Conv1d -> BatchNorm1d -> ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # 最后一层，不需要激活
        x = self.conv6(x)
        
        # 将输出转置回 [batch_size, T, 6]
        x = x.permute(0, 2, 1)
        
        return x

# 示例输入
batch_size, T = 4, 10  # 假设 batch_size 为 4，T 为 10
input_tensor = torch.randn(batch_size, T, 1)

# 创建模型实例并前向传播
model = MY_DEConv()
output_tensor = model(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")
