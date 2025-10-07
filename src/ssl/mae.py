"""Masked autoencoder-style self-supervised learning on the Iris dataset.

This script mimics the essence of MAE by randomly masking feature values and
training an autoencoder to reconstruct the missing values. After SSL training,
we freeze the encoder and fine-tune a supervised classifier head using the
labeled Iris targets.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MaskedAutoencoder(nn.Module):
    """Small MLP-based masked autoencoder for tabular data."""

    def __init__(self, n_features: int, latent_dim: int) -> None:
        super().__init__()
        hidden_width = max(16, n_features * 4)
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, n_features),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x.clone()
        x_masked[mask] = 0.0
        latent = self.encoder(x_masked)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


def masked_mse_loss(inputs: torch.Tensor, recon: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_count = mask.sum().item()
    if masked_count == 0:
        return nn.functional.mse_loss(recon, inputs)
    squared_error = (recon - inputs) ** 2
    return (squared_error * mask.float()).sum() / mask.float().sum()


def prepare_data(path: Path, test_size: float, seed: int):
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled.astype(np.float32),
        X_test_scaled.astype(np.float32),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def train_mae(
    model: MaskedAutoencoder,
    dataloader: DataLoader,
    epochs: int,
    mask_ratio: float,
    noise_std: float,
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in dataloader:
            clean_inputs = batch[0].to(device)
            inputs = clean_inputs
            if noise_std > 0:
                inputs = clean_inputs + torch.randn_like(clean_inputs) * noise_std
            mask = (torch.rand_like(inputs) < mask_ratio).to(device)

            optimizer.zero_grad()
            recon, _ = model(inputs, mask)
            loss = masked_mse_loss(clean_inputs, recon, mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            mean_loss = epoch_loss / len(dataloader.dataset)
            print(f"Epoch {epoch:03d}/{epochs} - masked MSE: {mean_loss:.4f}")


def fine_tune_with_frozen_encoder(
    model: MaskedAutoencoder,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train,
    y_test,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    noise_std: float,
) -> float:
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train).astype(np.int64)
    y_test_encoded = label_encoder.transform(y_test).astype(np.int64)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train_encoded)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test_encoded)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()

    head = nn.Linear(model.latent_dim, len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs_for_encoder = inputs
            if noise_std > 0:
                inputs_for_encoder = inputs + torch.randn_like(inputs) * noise_std

            with torch.no_grad():
                latent = model.encoder(inputs_for_encoder)
            logits = head(latent)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

        if epoch % max(1, epochs // 5) == 0 or epoch == 1:
            train_acc = total_correct / total_samples
            mean_loss = total_loss / total_samples
            print(
                f"Head epoch {epoch:03d}/{epochs} - loss: {mean_loss:.4f} - acc: {train_acc:.4f}"
            )

    head.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            latent = model.encoder(inputs)
            logits = head(latent)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if not all_preds:
        print("No test samples available for evaluation.")
        return 0.0

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    accuracy = accuracy_score(y_true, y_pred)
    mapping = {int(idx): label for idx, label in enumerate(label_encoder.classes_)}
    print(f"Frozen-encoder fine-tune accuracy: {accuracy:.4f}")
    print("Label mapping:", mapping)

    return accuracy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Masked autoencoder Iris example")
    parser.add_argument("--epochs", type=int, default=200, help="Number of SSL training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for MAE training")
    parser.add_argument("--mask-ratio", type=float, default=0.5, help="Probability of masking each feature")
    parser.add_argument("--latent-dim", type=int, default=8, help="Dimensionality of the latent representation")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test set proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data", type=Path, default=Path("iris.csv"), help="Path to the Iris CSV file")
    parser.add_argument("--ft-epochs", type=int, default=150, help="Supervised head training epochs")
    parser.add_argument("--ft-lr", type=float, default=5e-3, help="Learning rate for the classifier head")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.1,
        help="Std of Gaussian noise added during MAE training",
    )
    parser.add_argument(
        "--ft-noise-std",
        type=float,
        default=0.05,
        help="Std of Gaussian noise added when training the classifier head",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if not args.data.exists():
        raise FileNotFoundError(f"Could not find dataset at {args.data}")

    X_train, X_test, y_train, y_test = prepare_data(args.data, args.test_size, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedAutoencoder(n_features=X_train.shape[1], latent_dim=args.latent_dim).to(device)

    dataset = TensorDataset(torch.from_numpy(X_train))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print("Starting masked autoencoder training...")
    train_mae(
        model,
        dataloader,
        args.epochs,
        args.mask_ratio,
        args.noise_std,
        device,
    )

    print("Fine-tuning a frozen encoder with a supervised head...")
    fine_tune_with_frozen_encoder(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        device,
        epochs=args.ft_epochs,
        batch_size=args.batch_size,
        lr=args.ft_lr,
        noise_std=args.ft_noise_std,
    )


if __name__ == "__main__":
    main()
