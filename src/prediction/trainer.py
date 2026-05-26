import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .model import TrajectoryLSTM


class ModelManager:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = TrajectoryLSTM(
            hidden_size=config.get('hidden_size', 256),
            num_layers=config.get('num_layers', 3),
            pred_len=config.get('pred_len', 10),
            dropout=config.get('dropout', 0.2),
        ).to(device)
        self.model.eval()

    def load_pretrained(self, path):
        if Path(path).exists():
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            return True
        return False

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def predict(self, seq_tensor):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(seq_tensor.to(self.device))
        return pred.squeeze(0).cpu().numpy()  # (pred_len, 4)

    def predict_with_uncertainty(self, seq_tensor, n_samples: int = 20):
        """Returns (mean_pred, std_pred), each shape (pred_len, 4)."""
        mean, std = self.model.predict_with_uncertainty(
            seq_tensor.to(self.device), n_samples
        )
        return mean.squeeze(0).cpu().numpy(), std.squeeze(0).cpu().numpy()

    def finetune(self, sequences, targets, epochs=5, lr=0.001):
        if len(sequences) < 10:
            return
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        X = torch.tensor(np.array(sequences), dtype=torch.float32).to(self.device)
        Y = torch.tensor(np.array(targets), dtype=torch.float32).to(self.device)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(self.model(X), Y)
            loss.backward()
            optimizer.step()
        self.model.eval()


def pretrain_model(config, output_path):
    """Generate synthetic motion trajectories and pretrain the LSTM."""
    seq_len = config.get('seq_len', 20)
    pred_len = config.get('pred_len', 10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Pretrain] device={device}")

    def _linear(n=600):
        s, t = [], []
        for _ in range(n):
            cx, cy = np.random.uniform(0.1, 0.9, 2)
            w, h = np.random.uniform(0.05, 0.2, 2)
            vx, vy = np.random.uniform(-0.025, 0.025, 2)
            total = seq_len + pred_len
            traj = [[float(np.clip(cx + vx * i, 0, 1)),
                     float(np.clip(cy + vy * i, 0, 1)), w, h]
                    for i in range(total)]
            s.append(traj[:seq_len])
            t.append(traj[seq_len:])
        return s, t

    def _circular(n=500):
        s, t = [], []
        for _ in range(n):
            cx0, cy0 = np.random.uniform(0.3, 0.7, 2)
            r = np.random.uniform(0.05, 0.15)
            omega = np.random.uniform(0.05, 0.2) * np.random.choice([-1, 1])
            phase = np.random.uniform(0, 2 * np.pi)
            w, h = np.random.uniform(0.04, 0.12, 2)
            total = seq_len + pred_len
            traj = [[cx0 + r * np.cos(phase + omega * i),
                     cy0 + r * np.sin(phase + omega * i), w, h]
                    for i in range(total)]
            s.append(traj[:seq_len])
            t.append(traj[seq_len:])
        return s, t

    def _decel(n=400):
        s, t = [], []
        for _ in range(n):
            cx, cy = np.random.uniform(0.1, 0.9, 2)
            w, h = np.random.uniform(0.05, 0.2, 2)
            vx, vy = np.random.uniform(-0.03, 0.03, 2)
            decay = np.random.uniform(0.85, 0.98)
            total = seq_len + pred_len
            traj = []
            for _ in range(total):
                traj.append([float(np.clip(cx, 0, 1)),
                             float(np.clip(cy, 0, 1)), w, h])
                cx += vx; cy += vy; vx *= decay; vy *= decay
            s.append(traj[:seq_len])
            t.append(traj[seq_len:])
        return s, t

    all_s, all_t = [], []
    for fn in [_linear, _circular, _decel]:
        s, t = fn()
        all_s.extend(s)
        all_t.extend(t)

    X = torch.tensor(np.array(all_s, dtype=np.float32))
    Y = torch.tensor(np.array(all_t, dtype=np.float32))
    print(f"[Pretrain] {len(X)} samples")

    model = TrajectoryLSTM(
        hidden_size=config.get('hidden_size', 256),
        num_layers=config.get('num_layers', 3),
        pred_len=pred_len,
        dropout=config.get('dropout', 0.2),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    criterion = nn.MSELoss()
    epochs = config.get('pretrain_epochs', 50)
    bs = config.get('batch_size', 64)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        X, Y = X[perm], Y[perm]
        total_loss = 0.0
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].to(device)
            yb = Y[i:i + bs].to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"[Pretrain] epoch {epoch+1}/{epochs}  loss={total_loss:.4f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[Pretrain] saved → {output_path}")
