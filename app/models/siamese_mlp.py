import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#双塔共享编码器，处理用户特征
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super(MLPEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

#对两个用户embedding进行比较, 输出相似度
class SiameseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SiameseMLP, self).__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim)
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        diff = torch.abs(x1 - x2)
        mult = x1*x2    
        concat = torch.cat([x1, x2, diff, mult], dim=1)
        return self.comparator(concat)
        
