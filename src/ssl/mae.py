"""Masked autoencoder-style self-supervised learning on the Iris dataset.

This module trains a masked autoencoder (MAE) on the Iris features, stores the
weights for external analysis, and offers multiple downstream evaluation paths:
  * `linear`: logistic-regression probing on frozen encoder representations.
  * `ft`: supervised head training with the pretrained frozen encoder and a
    randomly initialised encoder baseline to highlight representation quality.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class CheckpointMetadata:
    """Lightweight configuration snapshot stored next to the weights."""

    latent_dim: int
    mask_ratio: float
    mae_epochs: int
    noise_std: float
    ft_epochs: int
    ft_noise_std: float
    seed: int


class MaskedAutoencoder(nn.Module):
    """Small MLP-based masked autoencoder suitable for tabular inputs."""

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def add_gaussian_noise(tensor: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return tensor
    return tensor + torch.randn_like(tensor) * std


def masked_mse_loss(inputs: torch.Tensor, recon: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_count = mask.sum().item()
    if masked_count == 0:
        return nn.functional.mse_loss(recon, inputs)
    squared_error = (recon - inputs) ** 2
    return (squared_error * mask.float()).sum() / mask.float().sum()


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
            noised_inputs = add_gaussian_noise(clean_inputs, noise_std)
            mask = (torch.rand_like(clean_inputs) < mask_ratio).to(device)

            optimizer.zero_grad()
            recon, _ = model(noised_inputs, mask)
            loss = masked_mse_loss(clean_inputs, recon, mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * clean_inputs.size(0)

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            mean_loss = epoch_loss / len(dataloader.dataset)
            print(f"Epoch {epoch:03d}/{epochs} - masked MSE: {mean_loss:.4f}")


def evaluate_with_logistic(
    encoder: nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train,
    y_test,
    device: torch.device,
    max_iter: int,
) -> float:
    encoder.eval()
    with torch.no_grad():
        train_latent = encoder(torch.from_numpy(X_train).to(device)).cpu().numpy()
        test_latent = encoder(torch.from_numpy(X_test).to(device)).cpu().numpy()

    clf = LogisticRegression(max_iter=max_iter, multi_class="auto")
    clf.fit(train_latent, y_train)
    y_pred = clf.predict(test_latent)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic regression accuracy: {accuracy:.4f}")
    return accuracy


def train_head_with_frozen_encoder(
    encoder: nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train,
    y_test,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    noise_std: float,
    tag: str = "pretrained",
    latent_dim: Optional[int] = None,
) -> float:
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

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

    if latent_dim is None:
        with torch.no_grad():
            latent_dim = encoder(torch.zeros(1, X_train.shape[1], device=device)).shape[1]
    head = nn.Linear(latent_dim, len(label_encoder.classes_)).to(device)
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

            inputs_for_encoder = add_gaussian_noise(inputs, noise_std)

            with torch.no_grad():
                latent = encoder(inputs_for_encoder)
            logits = head(latent)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

        if epoch % max(1, epochs // 5) == 0 or epoch == 1:
            train_acc = total_correct / max(1, total_samples)
            mean_loss = total_loss / max(1, total_samples)
            print(
                f"Head ({tag}) epoch {epoch:03d}/{epochs} - loss: {mean_loss:.4f} - acc: {train_acc:.4f}"
            )

    head.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            latent = encoder(inputs)
            logits = head(latent)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if not all_preds:
        print(f"No test samples available for {tag} evaluation.")
        return 0.0

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    accuracy = accuracy_score(y_true, y_pred)
    mapping = {int(idx): label for idx, label in enumerate(label_encoder.classes_)}
    print(f"Frozen-encoder ({tag}) accuracy: {accuracy:.4f}")
    print(f"Label mapping ({tag}): {mapping}")
    return accuracy


def save_checkpoint(model: MaskedAutoencoder, directory: Path, metadata: CheckpointMetadata) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), directory / "encoder.pt")
    torch.save(model.decoder.state_dict(), directory / "decoder.pt")
    with open(directory / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(metadata), fh, indent=2)
    print(f"Saved encoder/decoder weights to {directory}")


def load_checkpoint(model: MaskedAutoencoder, directory: Path, device: torch.device) -> bool:
    encoder_path = directory / "encoder.pt"
    decoder_path = directory / "decoder.pt"
    if not encoder_path.exists() or not decoder_path.exists():
        print(f"Checkpoint missing encoder/decoder weights under {directory}, skipping load.")
        return False

    encoder_state = torch.load(encoder_path, map_location=device)
    decoder_state = torch.load(decoder_path, map_location=device)
    model.encoder.load_state_dict(encoder_state)
    model.decoder.load_state_dict(decoder_state)
    print(f"Loaded encoder/decoder weights from {directory}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masked autoencoder Iris example")
    parser.add_argument("--epochs", type=int, default=200, help="Number of SSL training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for MAE training")
    parser.add_argument("--mask-ratio", type=float, default=0.75, help="Probability of masking each feature")
    parser.add_argument("--latent-dim", type=int, default=8, help="Dimensionality of the latent representation")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test set proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data", type=Path, default=Path("iris.csv"), help="Path to the Iris CSV file")
    parser.add_argument("--noise-std", type=float, default=0.1, help="Std of Gaussian noise added during MAE training")
    parser.add_argument("--ft-epochs", type=int, default=150, help="Supervised head training epochs")
    parser.add_argument("--ft-lr", type=float, default=5e-3, help="Learning rate for the classifier head")
    parser.add_argument("--ft-noise-std", type=float, default=0.05, help="Std of Gaussian noise added when training the classifier head")
    parser.add_argument("--logistic-max-iter", type=int, default=1000, help="Max iterations for logistic regression evaluation")
    parser.add_argument("--eval-mode", choices=["linear", "ft"], default="ft", help="Downstream evaluation strategy")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/mae"), help="Directory used to store model weights")
    parser.add_argument("--load-checkpoint", type=Path, help="Directory containing encoder.pt/decoder.pt to preload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    if not args.data.exists():
        raise FileNotFoundError(f"Could not find dataset at {args.data}")

    X_train, X_test, y_train, y_test = prepare_data(args.data, args.test_size, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedAutoencoder(n_features=X_train.shape[1], latent_dim=args.latent_dim).to(device)

    if args.load_checkpoint:
        load_checkpoint(model, args.load_checkpoint, device)

    if args.epochs > 0:
        dataset = TensorDataset(torch.from_numpy(X_train))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        print("Starting masked autoencoder training...")
        train_mae(model, dataloader, args.epochs, args.mask_ratio, args.noise_std, device)

    metadata = CheckpointMetadata(
        latent_dim=args.latent_dim,
        mask_ratio=args.mask_ratio,
        mae_epochs=args.epochs,
        noise_std=args.noise_std,
        ft_epochs=args.ft_epochs,
        ft_noise_std=args.ft_noise_std,
        seed=args.seed,
    )
    save_checkpoint(model, args.checkpoint_dir, metadata)

    print(f"Evaluation mode: {args.eval_mode}")
    if args.eval_mode == "linear":
        evaluate_with_logistic(
            model.encoder,
            X_train,
            X_test,
            y_train,
            y_test,
            device,
            max_iter=args.logistic_max_iter,
        )
        return

    # Fine-tuning path: compare pretrained encoder vs randomly initialised baseline.
    pretrained_accuracy = train_head_with_frozen_encoder(
        model.encoder,
        X_train,
        X_test,
        y_train,
        y_test,
        device,
        epochs=args.ft_epochs,
        batch_size=args.batch_size,
        lr=args.ft_lr,
        noise_std=args.ft_noise_std,
        tag="pretrained",
        latent_dim=model.latent_dim,
    )

    set_seed(args.seed + 1)  # ensure different random init from pretrained path
    baseline_model = MaskedAutoencoder(n_features=X_train.shape[1], latent_dim=args.latent_dim).to(device)
    baseline_accuracy = train_head_with_frozen_encoder(
        baseline_model.encoder,
        X_train,
        X_test,
        y_train,
        y_test,
        device,
        epochs=args.ft_epochs,
        batch_size=args.batch_size,
        lr=args.ft_lr,
        noise_std=args.ft_noise_std,
        tag="random-init",
        latent_dim=baseline_model.latent_dim,
    )

    print(
        "\nSummary:"
        f"\n  Pretrained encoder accuracy: {pretrained_accuracy:.4f}"
        f"\n  Random-init encoder accuracy: {baseline_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
