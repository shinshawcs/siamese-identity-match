import torch
import torch.nn as nn

class MLPEncoder_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super(MLPEncoder_v2, self).__init__()
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

class SiameseMLP_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(SiameseMLP_v2, self).__init__()
        self.encoder = MLPEncoder_v2(input_dim, hidden_dim, dropout=0.3)
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        diff = torch.abs(x1 - x2)
        mult = x1 * x2
        concat = torch.cat([x1, x2, diff, mult], dim=1)
        return self.comparator(concat)