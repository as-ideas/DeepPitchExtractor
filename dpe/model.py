from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNormConv(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel,
                              padding=kernel // 2)
        self.bnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.relu(x)
        x = self.bnorm(x)
        return x


class PitchExtractor(torch.nn.Module):

    def __init__(self,
                 spec_dim: int,
                 n_channels: int,
                 conv_dim: int = 256,
                 dropout: float = 0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            BatchNormConv(spec_dim, conv_dim, 5),
            BatchNormConv(conv_dim, conv_dim, 5),
            BatchNormConv(conv_dim, conv_dim, 5),
        ])
        self.logit_lin = nn.Linear(conv_dim, n_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.transpose(1, 2)
        logit_out = self.logit_lin(x).transpose(1, 2)
        return logit_out