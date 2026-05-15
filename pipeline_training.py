"""
Standalone IELTS AES training pipeline.

This script contains the training code needed to load IELTS essay data, fine-tune
a transformer regressor with an extra numerical feature, save the best local
checkpoint, and print validation/test metrics.
"""

from __future__ import annotations

import math
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel, DebertaModel, DebertaV2Model, RobertaModel


CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config() -> Dict[str, Any]:
    """Load base config, merge the selected training config, then apply env overrides."""
    config = _load_yaml(CONFIG_DIR / "base.yaml")
    method = os.getenv("DATA_AUGMENTATION_METHOD", config.get("data_augmentation_method", "training_worst_band"))
    method_path = CONFIG_DIR / f"{method}.yaml"
    if method_path.exists():
        config.update(_load_yaml(method_path))
    elif method != "base":
        print(f"Warning: {method_path} not found, using base config only")

    env_overrides = {
        "MAX_SEQ_LENGTH": ("max_seq_length", int),
        "BERT_MODEL_NAME": ("bert_model_name", str),
        "LEARNING_RATE": ("learning_rate", float),
        "NUM_FREEZE_LAYERS": ("num_freeze_layers", int),
        "DROPOUT_RATE": ("dropout_rate", float),
        "BATCH_SIZE": ("batch_size", int),
        "EPOCHS": ("epochs", int),
        "LOSS": ("loss", str),
        "PATIENCE": ("patience", int),
        "USE_SIGMOID": ("use_sigmoid", _parse_bool),
        "GRADIENT_ACCUMULATION_STEPS": ("gradient_accumulation_steps", int),
        "USE_MIXED_PRECISION": ("use_mixed_precision", _parse_bool),
        "GRADIENT_CHECKPOINTING": ("gradient_checkpointing", _parse_bool),
        "DATA_PATH": ("data_path", str),
        "DATA_FOLDER": ("data_folder", str),
        "TRAIN_FOLDER": ("train_folder", str),
        "VALIDATION_FOLDER": ("validation_folder", str),
        "TEST_FOLDER": ("test_folder", str),
        "MODEL_SAVE_PATH": ("model_save_path", str),
        "SCALER_PATH": ("scaler_path", str),
        "CHECKPOINT_PATH": ("checkpoint_path", str),
        "DEVICE": ("device", str),
        "MIN_SCORE": ("min_score", float),
        "MAX_SCORE": ("max_score", float),
        "DATA_AUGMENTATION_METHOD": ("data_augmentation_method", str),
    }
    for env_key, (config_key, parser) in env_overrides.items():
        if env_key in os.environ:
            config[config_key] = parser(os.environ[env_key])

    return config


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_timestamp(folder_name: str) -> datetime:
    timestamp_str = folder_name.rsplit("_", 1)[-1]
    return datetime.strptime(timestamp_str, "%y-%m-%d-%H-%M")


def find_latest_data_folder(base_path: str) -> str:
    folders = []
    for name in os.listdir(base_path):
        full_path = os.path.join(base_path, name)
        if os.path.isdir(full_path) and "_" in name:
            try:
                extract_timestamp(name)
                folders.append(name)
            except ValueError:
                continue
    if not folders:
        raise ValueError(f"No versioned folders found in {base_path}")
    return os.path.join(base_path, max(folders, key=extract_timestamp))


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"question": "Question", "answer": "Essay"})
    if "Essay Length" not in df.columns:
        df["Essay Length"] = df["Essay"].apply(lambda x: len(str(x).split()) if pd.notna(x) else None)
    if "Question Length" not in df.columns:
        df["Question Length"] = df["Question"].apply(lambda x: len(str(x).split()) if pd.notna(x) else None)
    if "Length" not in df.columns:
        df["Length"] = df["Essay"].apply(lambda x: len(str(x)) if pd.notna(x) else None)
    return df[df["Overall"] >= 5]


def read_and_concat_csv(folder_path: str) -> pd.DataFrame:
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder_path}")
    frames = []
    for file_name in csv_files:
        frames.append(process_dataframe(pd.read_csv(os.path.join(folder_path, file_name))))
    return pd.concat(frames, ignore_index=True)


def load_dataset_splits(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.get("data_folder"):
        data_root = os.path.join(config["data_path"], config["data_folder"])
    else:
        data_root = find_latest_data_folder(config["data_path"])
    train_df = read_and_concat_csv(os.path.join(data_root, config["train_folder"]))
    val_df = read_and_concat_csv(os.path.join(data_root, config["validation_folder"]))
    test_df = read_and_concat_csv(os.path.join(data_root, config["test_folder"]))
    return train_df, val_df, test_df


def extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    questions = np.array(df["Question"])
    essays = np.array(df["Essay"])
    lengths = np.array(df["Length"]).reshape(-1, 1).astype(np.float32)
    scores = np.array(df["Overall"]).astype(np.float32)
    return questions, essays, lengths, scores


def tokenize_inputs(questions, essays, scores, tokenizer, max_length: int, print_stats: bool = False) -> Dict[str, torch.Tensor]:
    input_ids_list = []
    attention_masks_list = []
    lengths_token = []
    for question, essay, _score in zip(questions, essays, scores):
        encoding = tokenizer(
            question,
            essay,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids_list.append(encoding["input_ids"].squeeze(0))
        attention_masks_list.append(encoding["attention_mask"].squeeze(0))
        lengths_token.append(len(encoding["input_ids"][0]))

    if print_stats and lengths_token:
        print(f"  Max token length: {max(lengths_token)}")
        print(f"  Min token length: {min(lengths_token)}")
        print(f"  Mean token length: {np.mean(lengths_token):.1f}")

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_masks_list),
    }


def create_dataloaders(
    questions_train,
    essays_train,
    lengths_train,
    scores_train,
    questions_val,
    essays_val,
    lengths_val,
    scores_val,
    questions_test,
    essays_test,
    lengths_test,
    scores_test,
    tokenizer,
    batch_size: int,
    max_seq_length: int,
    val_batch_size: int = 1,
    test_batch_size: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    print("\nTokenizing data...")
    print("Training data...")
    train_encodings = tokenize_inputs(
        list(questions_train), list(essays_train), list(scores_train), tokenizer, max_seq_length, print_stats=True
    )
    print("Validation data...")
    val_encodings = tokenize_inputs(list(questions_val), list(essays_val), list(scores_val), tokenizer, max_seq_length)
    print("Test data...")
    test_encodings = tokenize_inputs(list(questions_test), list(essays_test), list(scores_test), tokenizer, max_seq_length)

    train_dataset = TensorDataset(
        train_encodings["input_ids"],
        train_encodings["attention_mask"],
        torch.tensor(lengths_train, dtype=torch.float32),
        torch.tensor(scores_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        val_encodings["input_ids"],
        val_encodings["attention_mask"],
        torch.tensor(lengths_val, dtype=torch.float32),
        torch.tensor(scores_val, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        test_encodings["input_ids"],
        test_encodings["attention_mask"],
        torch.tensor(lengths_test, dtype=torch.float32),
        torch.tensor(scores_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, drop_last=True)

    print("\n" + "=" * 80)
    print("DATALOADER SUMMARY")
    print("=" * 80)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    return train_loader, val_loader, test_loader


class TransformerWithExtraFeature(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        use_sigmoid: bool = True,
        dropout_prob: float = 0.155934085175111,
        num_trainable_layers: int = 4,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        model_name_lower = pretrained_model_name.lower()
        if "deberta" in model_name_lower:
            if "v3" in model_name_lower or "deberta-v2" in model_name_lower:
                self.transformer = DebertaV2Model.from_pretrained(pretrained_model_name, use_safetensors=True)
            else:
                self.transformer = DebertaModel.from_pretrained(pretrained_model_name, use_safetensors=True)
        elif "roberta" in model_name_lower:
            self.transformer = RobertaModel.from_pretrained(pretrained_model_name, use_safetensors=True)
        elif "bert" in model_name_lower:
            self.transformer = BertModel.from_pretrained(pretrained_model_name, use_safetensors=True)
        else:
            self.transformer = AutoModel.from_pretrained(pretrained_model_name, use_safetensors=True)

        self.use_sigmoid = use_sigmoid
        self.model_type = model_name_lower
        hidden_size = self.transformer.config.hidden_size

        for param in self.transformer.parameters():
            param.requires_grad = False

        encoder_layers = getattr(self.transformer.encoder, "layer", None)
        if encoder_layers is None:
            raise AttributeError("Expected transformer.encoder.layer to exist for layer freezing")
        for layer in encoder_layers[-num_trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        if gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print("Gradient checkpointing: enabled")

        self.num_feature_norm = nn.BatchNorm1d(1)
        self.fc0 = nn.Linear(hidden_size + 1, 512)
        self.relu0 = nn.ReLU()
        self.dropout0 = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(512, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_prob)
        self.output_layer = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, extra_number: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        if "deberta" in self.model_type:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        else:
            pooled_output = outputs.pooler_output

        if extra_number.dim() == 1:
            extra_number = extra_number.unsqueeze(1)
        normalized_num = self.num_feature_norm(extra_number)
        x = torch.cat((pooled_output, normalized_num), dim=1)
        x = self.dropout0(self.relu0(self.fc0(x)))
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        raw_output = self.output_layer(x)
        if self.use_sigmoid:
            return 4 + 5 * self.sigmoid(raw_output)
        return raw_output


def print_model_summary(model: nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Total Parameters:         {total_params:,}")
    print(f"Trainable Parameters:     {trainable_params:,}")
    print(f"Non-Trainable Parameters: {total_params - trainable_params:,}")
    print("=" * 60)
    print(model)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().reshape(-1)


def train_fn(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    device,
    gradient_accumulation_steps: int = 1,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    n_samples = 0
    use_amp = grad_scaler is not None

    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Train", unit="batch", leave=False)
    for step, (input_ids, attention_mask, extra_number, labels) in enumerate(pbar):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        extra_number = extra_number.to(device)
        labels = labels.to(device).unsqueeze(1)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(input_ids, attention_mask, extra_number)
            loss = criterion(outputs, labels) / gradient_accumulation_steps

        if use_amp:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(loader):
            if use_amp:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        batch_size = labels.shape[0]
        n_samples += batch_size
        total_loss += loss.item() * gradient_accumulation_steps * batch_size
        y_true = _to_numpy(labels)
        y_pred = _to_numpy(outputs)
        total_mae += mean_absolute_error(y_true, y_pred) * batch_size
        total_rmse += math.sqrt(mean_squared_error(y_true, y_pred)) * batch_size

    return {
        "loss": total_loss / max(n_samples, 1),
        "mae": total_mae / max(n_samples, 1),
        "rmse": total_rmse / max(n_samples, 1),
        "n": n_samples,
    }


def round_ielts_bands(scores: np.ndarray) -> np.ndarray:
    rounded = []
    for score in np.array(scores, dtype=float):
        frac = score - np.floor(score)
        base = np.floor(score)
        if frac < 0.25:
            rounded.append(base)
        elif frac < 0.75:
            rounded.append(base + 0.5)
        else:
            rounded.append(base + 1.0)
    return np.array(rounded)


@torch.no_grad()
def valid_fn(model: nn.Module, loader: DataLoader, criterion, device, use_amp: bool = False) -> Tuple[Dict[str, float], Dict]:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    y_true_all = []
    y_pred_all = []

    for input_ids, attention_mask, extra_number, labels in tqdm(loader, desc="Val", unit="batch", leave=False):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        extra_number = extra_number.to(device)
        labels = labels.to(device).unsqueeze(1)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(input_ids, attention_mask, extra_number)
            loss = criterion(outputs, labels)

        batch_size = labels.shape[0]
        n_samples += batch_size
        total_loss += loss.item() * batch_size
        y_true_all.append(_to_numpy(labels))
        y_pred_all.append(_to_numpy(outputs))

    if not y_true_all:
        return {"loss": 0.0, "mae": 0.0, "rmse": 0.0, "n": 0}, {}

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    mse = mean_squared_error(y_true_all, y_pred_all)
    return {
        "loss": total_loss / max(n_samples, 1),
        "mae": mean_absolute_error(y_true_all, y_pred_all),
        "rmse": math.sqrt(mse),
        "n": n_samples,
    }, {}


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_int = (np.array(y_true) * 2).astype(int)
    y_pred_int = (np.array(y_pred) * 2).astype(int)
    return cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic")


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_pred_rounded = round_ielts_bands(y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred_rounded),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred_rounded)),
        "mape": mean_absolute_percentage_error(y_true, y_pred_rounded),
        "qwk": quadratic_weighted_kappa(y_true, y_pred_rounded),
        "mse": mean_squared_error(y_true, y_pred_rounded),
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "std_true": float(np.std(y_true)),
        "std_pred": float(np.std(y_pred)),
    }


def compute_per_band_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Compute per-band metrics using the same raw-prediction logic as the source repo."""
    if len(y_true) == 0:
        return pd.DataFrame(columns=["band", "count", "mse", "mae", "rmse", "mean_pred", "mean_true"])

    frame = pd.DataFrame(
        {
            "band": round_ielts_bands(y_true),
            "target": np.array(y_true, dtype=float),
            "prediction": np.array(y_pred, dtype=float),
        }
    )
    rows = []
    for band, group in frame.groupby("band", sort=True):
        mse = mean_squared_error(group["target"], group["prediction"])
        rows.append(
            {
                "band": float(band),
                "count": int(len(group)),
                "mse": mse,
                "mae": mean_absolute_error(group["target"], group["prediction"]),
                "rmse": math.sqrt(mse),
                "mean_pred": float(group["prediction"].mean()),
                "mean_true": float(group["target"].mean()),
            }
        )
    return pd.DataFrame(rows)


def print_per_band_metrics(metrics_df: pd.DataFrame, split_name: str) -> None:
    print(f"\n{split_name} per-band metrics:")
    if metrics_df.empty:
        print("  No samples available.")
        return
    display_df = metrics_df.copy()
    for column in ["mse", "mae", "rmse", "mean_pred", "mean_true"]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")
    print(display_df.to_string(index=False))


def setup_device_and_config() -> Tuple[torch.device, Dict[str, Any]]:
    print("=" * 80)
    print("IELTS AES Training Pipeline")
    print("=" * 80)
    seed_everything(seed=42)
    config = load_config()
    requested_device = str(config.get("device", "cuda"))
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)
    print(f"\nDevice: {device}")
    print(f"Model: {config['bert_model_name']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    return device, config


def load_and_prepare_data(config: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    print(f"Data path: {config['data_path']}")
    print(f"Data folder: {config.get('data_folder')}")
    print(f"Train folder: {config['train_folder']}")

    train_df, val_df, test_df = load_dataset_splits(config)
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    if len(train_df) % config["batch_size"] == 1:
        train_df = train_df.iloc[1:].reset_index(drop=True)

    q_train, e_train, len_train, y_train = extract_features(train_df)
    q_val, e_val, len_val, y_val = extract_features(val_df)
    q_test, e_test, len_test, y_test = extract_features(test_df)
    return (
        train_df,
        val_df,
        test_df,
        q_train,
        e_train,
        len_train,
        y_train,
        q_val,
        e_val,
        len_val,
        y_val,
        q_test,
        e_test,
        len_test,
        y_test,
    )


def normalize_features(len_train, len_val, len_test, config: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("NORMALIZING NUMERICAL FEATURES")
    print("=" * 80)
    scaler = StandardScaler()
    len_train_std = scaler.fit_transform(len_train)
    len_val_std = scaler.transform(len_val)
    len_test_std = scaler.transform(len_test)

    scaler_path = Path(config["scaler_path"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    return len_train_std, len_val_std, len_test_std


def initialize_model(device, config: Dict[str, Any]) -> nn.Module:
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    model = TransformerWithExtraFeature(
        pretrained_model_name=config["bert_model_name"],
        use_sigmoid=config["use_sigmoid"],
        dropout_prob=config["dropout_rate"],
        num_trainable_layers=config["num_freeze_layers"],
        gradient_checkpointing=config.get("gradient_checkpointing", False),
    )
    model.to(device)
    print_model_summary(model)
    return model


def setup_training(model: nn.Module, config: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("TRAINING SETUP")
    print("=" * 80)
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    use_amp = bool(config.get("use_mixed_precision", False)) and torch.cuda.is_available()
    grad_scaler = torch.amp.GradScaler("cuda") if use_amp else None
    grad_accum = int(config.get("gradient_accumulation_steps", 1))
    if use_amp:
        print("Mixed precision enabled")
    if grad_accum > 1:
        print(f"Gradient accumulation: {grad_accum} steps")
    return optimizer, criterion, grad_scaler, grad_accum


def training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    criterion,
    device,
    config: Dict[str, Any],
    grad_scaler=None,
    gradient_accumulation_steps: int = 1,
) -> Tuple[list[float], list[float]]:
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = Path(config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["epochs"]):
        print(f"\n[Epoch {epoch + 1}/{config['epochs']}]")
        train_metrics = train_fn(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grad_scaler=grad_scaler,
        )
        print(
            f"  Train Loss: {train_metrics['loss']:.4f} | "
            f"MAE: {train_metrics['mae']:.4f} | RMSE: {train_metrics['rmse']:.4f}"
        )
        train_losses.append(train_metrics["loss"])

        val_metrics, _ = valid_fn(model, val_loader, criterion, device, use_amp=(grad_scaler is not None))
        print(f"  Val Loss: {val_metrics['loss']:.4f} | MAE: {val_metrics['mae']:.4f} | RMSE: {val_metrics['rmse']:.4f}")
        val_losses.append(val_metrics["loss"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_metrics["loss"],
                "mae": val_metrics["mae"],
                "rmse": val_metrics["rmse"],
                "config": config,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  Best model saved to {checkpoint_path} (loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config['patience']})")

        if patience_counter >= config["patience"]:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return train_losses, val_losses


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device, split_name: str, min_score: float, max_score: float):
    print("\n" + "=" * 80)
    print(f"{split_name.upper()} EVALUATION")
    print("=" * 80)
    model.eval()
    all_preds = []
    all_trues = []
    for input_ids, attention_mask, extra_num, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        extra_num = extra_num.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, attention_mask, extra_num)
        outputs = torch.clamp(outputs, min=min_score, max=max_score)
        all_preds.extend(outputs.squeeze(1).cpu().numpy())
        all_trues.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    metrics = compute_metrics(all_trues, all_preds)
    print(f"{split_name} MAE: {metrics['mae']:.4f}")
    print(f"{split_name} RMSE: {metrics['rmse']:.4f}")
    print(f"{split_name} QWK: {metrics['qwk']:.4f}")
    print(f"{split_name} MAPE: {metrics['mape']:.4f}")
    print(f"{split_name} MSE: {metrics['mse']:.4f}")
    per_band_metrics = compute_per_band_metrics(all_trues, all_preds)
    print_per_band_metrics(per_band_metrics, split_name)
    if split_name.lower() in {"val", "validation"}:
        per_band_path = Path("outputs/metrics/per_band_val_metrics.csv")
        per_band_path.parent.mkdir(parents=True, exist_ok=True)
        per_band_metrics.to_csv(per_band_path, index=False)
        print(f"{split_name} per-band metrics saved to {per_band_path}")
    return all_trues, all_preds, metrics


def main() -> None:
    device, config = setup_device_and_config()
    (
        _train_df,
        _val_df,
        _test_df,
        q_train,
        e_train,
        len_train,
        y_train,
        q_val,
        e_val,
        len_val,
        y_val,
        q_test,
        e_test,
        len_test,
        y_test,
    ) = load_and_prepare_data(config)
    len_train_std, len_val_std, len_test_std = normalize_features(len_train, len_val, len_test, config)

    tokenizer = AutoTokenizer.from_pretrained(config["bert_model_name"])
    train_loader, val_loader, test_loader = create_dataloaders(
        q_train,
        e_train,
        len_train_std,
        y_train,
        q_val,
        e_val,
        len_val_std,
        y_val,
        q_test,
        e_test,
        len_test_std,
        y_test,
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_seq_length=config["max_seq_length"],
        val_batch_size=1,
        test_batch_size=1,
    )

    model = initialize_model(device, config)
    optimizer, criterion, grad_scaler, grad_accum = setup_training(model, config)
    training_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        config,
        grad_scaler=grad_scaler,
        gradient_accumulation_steps=grad_accum,
    )

    checkpoint = torch.load(config["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")

    min_score = float(config.get("min_score", 4.0))
    max_score = float(config.get("max_score", 9.0))
    evaluate_loader(model, val_loader, device, "Val", min_score, max_score)
    evaluate_loader(model, test_loader, device, "Test", min_score, max_score)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
