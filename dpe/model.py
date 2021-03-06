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


class PitchModel(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_channels: int = 32,
                 dropout: float = 0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            BatchNormConv(in_channels, conv_channels, 3),
            BatchNormConv(conv_channels, conv_channels, 3),
            BatchNormConv(conv_channels, conv_channels, 3),
        ])
        self.logit_lin = nn.Linear(conv_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.transpose(1, 2)
        logit_out = self.logit_lin(x).transpose(1, 2)
        return logit_out