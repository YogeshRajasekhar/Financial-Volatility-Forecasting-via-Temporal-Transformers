import math
from typing import Tuple
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEQ_LEN = 30
FORECAST_HORIZON = 1
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 15
LR = 1e-3
BATCH_SIZE = 32


def generate_financial_data(n_steps: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    mu = 0.0002
    omega, alpha, beta = 1e-6, 0.08, 0.9
    returns = np.zeros(n_steps, dtype=np.float32)
    sigma2 = np.full(n_steps, omega / (1 - alpha - beta), dtype=np.float32)

    for t in range(1, n_steps):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = mu + math.sqrt(sigma2[t]) * rng.standard_normal()

    returns_series = returns.astype(np.float32)
    realized_vol = np.zeros(n_steps, dtype=np.float32)
    window = 5
    for t in range(window, n_steps):
        realized_vol[t] = np.std(returns_series[t - window:t])

    return returns_series, realized_vol


def evaluate_garch_baseline(returns: np.ndarray, target_volatility: np.ndarray) -> Tuple[float, float]:
    split = int(len(returns) * 0.8)
    train_returns = returns[:split]
    test_len = len(returns) - split
    forecasts = np.zeros(test_len, dtype=np.float32)

    try:
        from arch import arch_model

        am = arch_model(train_returns * 100.0, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
        res = am.fit(disp="off")
        fc = res.forecast(horizon=test_len, reindex=False)
        variances = fc.variance.values[-1] / 10000.0
        forecasts = np.sqrt(np.maximum(variances, 1e-12)).astype(np.float32)
    except Exception:
        window = 20
        for i in range(test_len):
            idx = split + i
            lookback = returns[max(0, idx - window):idx]
            forecasts[i] = np.std(lookback) if len(lookback) > 0 else 0.0

    actual = target_volatility[split:split + test_len]
    mse = float(np.mean((forecasts - actual) ** 2))
    mae = float(np.mean(np.abs(forecasts - actual)))
    return mse, mae


class VolatilityDataset(Dataset):
    def __init__(self, returns: np.ndarray, target_volatility: np.ndarray,
                 seq_len: int = SEQ_LEN, horizon: int = FORECAST_HORIZON) -> None:
        self.seq_len = seq_len
        self.horizon = horizon

        rolling_vol_5 = np.zeros_like(returns)
        for t in range(5, len(returns)):
            rolling_vol_5[t] = np.std(returns[t - 5:t])

        features = np.stack([returns, rolling_vol_5], axis=-1).astype(np.float32)
        self.features = features
        self.targets = target_volatility.astype(np.float32)

        self.valid_starts = list(range(seq_len, len(returns) - horizon + 1))

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int):
        end = self.valid_starts[idx]
        start = end - self.seq_len
        x = self.features[start:end]
        y = self.targets[end + self.horizon - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                              (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TemporalTransformerModel(nn.Module):
    def __init__(self, d_in: int, d_model: int = D_MODEL, nhead: int = NHEAD,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        last_step = x[:, -1, :]
        out = self.output_head(last_step)
        return out.squeeze(-1)


def train_and_evaluate(returns: np.ndarray, target_volatility: np.ndarray) -> Tuple[float, float, float]:
    split = int(len(returns) * 0.8)

    train_returns, test_returns = returns[:split], returns[split - SEQ_LEN:]
    train_vol, test_vol = target_volatility[:split], target_volatility[split - SEQ_LEN:]

    train_ds = VolatilityDataset(train_returns, train_vol)
    test_ds = VolatilityDataset(test_returns, test_vol)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransformerModel(d_in=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train MSE: {epoch_loss / len(train_ds):.6f}")

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            all_preds.append(model(x_batch.to(device)).cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_preds, all_targets = np.concatenate(all_preds), np.concatenate(all_targets)

    mse = float(np.mean((all_preds - all_targets) ** 2))
    mae = float(np.mean(np.abs(all_preds - all_targets)))
    rmse = float(math.sqrt(mse))
    return mse, mae, rmse


if __name__ == "__main__":
    returns, target_volatility = generate_financial_data(n_steps=2000)

    garch_mse, garch_mae = evaluate_garch_baseline(returns, target_volatility)
    print(f"GARCH(1,1) Baseline - MSE: {garch_mse:.6f}, MAE: {garch_mae:.6f}")

    tf_mse, tf_mae, tf_rmse = train_and_evaluate(returns, target_volatility)
    print(f"Transformer - MSE: {tf_mse:.6f}, MAE: {tf_mae:.6f}, RMSE: {tf_rmse:.6f}")

    print("\nComparison Summary:")
    print(f"{'Model':<20}{'MSE':<12}{'MAE':<12}")
    print(f"{'GARCH(1,1)':<20}{garch_mse:<12.6f}{garch_mae:<12.6f}")
    print(f"{'Transformer':<20}{tf_mse:<12.6f}{tf_mae:<12.6f}")
