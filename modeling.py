from __future__ import annotations

import numpy as np
import torch
from lightning.pytorch import Trainer
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.linear_model import Lasso, Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import scale_frames


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_linear_model(
    train_df,
    val_df,
    feature_cols: list[str],
    target_col: str,
    model_type: str,
    scaler_kind: str,
    alpha: float,
):
    train_scaled, val_scaled, scaler = scale_frames(train_df, val_df, feature_cols, scaler_kind)
    if model_type == "ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "lasso":
        model = Lasso(alpha=alpha, max_iter=10000)
    else:
        raise ValueError(f"unknown linear model: {model_type}")
    model.fit(train_scaled[feature_cols], train_scaled[target_col])
    val_pred = model.predict(val_scaled[feature_cols])
    return model, scaler, val_pred


def predict_linear(model, scaler, df, feature_cols: list[str]) -> np.ndarray:
    scaled = scaler.transform(df[feature_cols])
    return model.predict(scaled)


class RNNRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        rnn_type: str,
    ) -> None:
        super().__init__()
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        last = output[:, -1, :]
        return self.head(last).squeeze(-1)


def train_rnn_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    rnn_type: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    lr: float,
    batch_size: int,
    epochs: int,
    seed: int,
) -> RNNRegressor:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNRegressor(
        input_size=train_x.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type=rnn_type,
    ).to(device)

    train_ds = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(val_x, dtype=torch.float32),
        torch.tensor(val_y, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                _ = loss_fn(model(xb), yb)
    return model


def predict_rnn(model: RNNRegressor, x: np.ndarray, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    preds = []
    loader = DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def train_tft_model(
    training,
    validation,
    hidden_size: int,
    hidden_continuous_size: int,
    dropout: float,
    attention_head_size: int,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    seed: int,
):
    set_seed(seed)
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(),
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        enable_checkpointing=False,
        logger=False,
        deterministic=True,
    )
    trainer.fit(tft, train_loader, val_loader)
    return tft


def predict_tft(model, dataset, batch_size: int):
    loader = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    preds, index = model.predict(loader, return_index=True)
    return preds.squeeze(-1), index
