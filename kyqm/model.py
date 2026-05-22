from __future__ import annotations

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.energy = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, outputs: torch.Tensor, last_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(last_hidden).unsqueeze(1)
        key = self.key(outputs)
        score = self.energy(torch.tanh(query + key)).squeeze(-1)
        weights = torch.softmax(score, dim=1)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        return context, weights


class LSTMAttentionRegressor(nn.Module):
    def __init__(
        self, *, input_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.attention = AdditiveAttention(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        context, attention_weights = self.attention(outputs, last_hidden)
        features = torch.cat([context, last_hidden], dim=1)
        prediction = self.head(features)
        return prediction, attention_weights
