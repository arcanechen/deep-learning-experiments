"""Shared direct multi-horizon architecture: change only the recurrent cell type."""
import torch
from torch import nn
from torch.utils.data import Dataset


class WeatherWindows(Dataset):
    def __init__(self, values, starts, target_index, input_hours=168, horizon=168):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.starts = starts
        self.target_index = target_index
        self.input_hours, self.horizon = input_hours, horizon

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, item):
        start = int(self.starts[item])
        middle = start + self.input_hours
        return (self.values[start:middle],
                self.values[middle:middle+self.horizon, self.target_index])


class WeatherForecaster(nn.Module):
    def __init__(self, cell, input_features=19, hidden_size=64, horizon=168):
        super().__init__()
        recurrent = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[cell]
        self.recurrent = recurrent(input_features, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        outputs, _ = self.recurrent(x)
        return self.head(outputs[:, -1, :])
