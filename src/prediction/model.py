import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryLSTM(nn.Module):
    """3-layer stacked LSTM for trajectory prediction.

    Ported from Keras-LSTM-Trajectory-Prediction to PyTorch.
    Input : (batch, seq_len, 4)  — [cx, cy, w, h] normalised to [0, 1]
    Output: (batch, pred_len, 4)
    """

    def __init__(self, input_size=4, hidden_size=256, num_layers=3,
                 pred_len=10, dropout=0.2):
        super().__init__()
        self.pred_len = pred_len
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, input_size * pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out.view(-1, self.pred_len, self.input_size)

    def predict_with_uncertainty(self, x, n_samples: int = 20):
        """MC-Dropout uncertainty estimation: run n forward passes with dropout active."""
        was_training = self.training
        self.train()
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(self.forward(x))
        if not was_training:
            self.eval()
        stacked = torch.stack(samples, dim=0)   # (n_samples, batch, pred_len, 4)
        return stacked.mean(dim=0), stacked.std(dim=0)
