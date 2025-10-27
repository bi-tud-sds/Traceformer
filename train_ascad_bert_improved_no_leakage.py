import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import h5py
import wandb
from tqdm import tqdm
import random
import math
from pathlib import Path

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set up device and tokenizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_ascad_data_with_attack_split(file_path, downsample=None, val_size=0.3, random_state=42):
    """
    Load ASCAD dataset with proper attack set splitting to prevent data leakage.

    Args:
        file_path: Path to ASCAD HDF5 file
        downsample: Optional downsampling of profiling set
        val_size: Fraction of attack set to use for validation (0.0-1.0)
        random_state: Random seed for splitting

    Returns:
        (train_data, val_data), normalization_stats, val_indices, test_indices
    """
    print(f"Loading dataset: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    with h5py.File(file_path, 'r') as in_file:
        # Load profiling (training) traces and metadata
        X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
        Y_profiling = np.array(in_file['Profiling_traces/labels'])
        plaintexts_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'])

        # Load attack traces and metadata
        X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)
        Y_attack = np.array(in_file['Attack_traces/labels'])
        plaintexts_attack = np.array(in_file['Attack_traces/metadata'][:]['plaintext'])

        # Normalize using profiling statistics (before any splitting)
        mean = np.mean(X_profiling, axis=0, keepdims=True)
        std = np.std(X_profiling, axis=0, keepdims=True)
        std[std == 0] = 1e-8  # Handle zero std

        X_profiling = (X_profiling - mean) / std
        X_attack = (X_attack - mean) / std

        # Optionally downsample the profiling set
        if downsample is not None:
            indices = np.random.choice(len(X_profiling), downsample, replace=False)
            X_profiling = X_profiling[indices]
            Y_profiling = Y_profiling[indices]
            plaintexts_profiling = plaintexts_profiling[indices]

    # Split attack set into validation and test with SAVED INDICES
    print(f"📊 Splitting attack set: {val_size:.1%} for validation, {1-val_size:.1%} reserved for testing")

    # Create indices for the attack set
    attack_indices = np.arange(len(X_attack))

    # Split indices (not the data directly) to track what goes where
    val_indices, test_indices = train_test_split(
        attack_indices,
        test_size=1-val_size, 
        random_state=random_state, 
        stratify=Y_attack
    )

    # Create validation data using validation indices
    X_val = X_attack[val_indices]
    Y_val = Y_attack[val_indices]
    plaintexts_val = plaintexts_attack[val_indices]

    # Training data (all profiling data)
    train_data = (X_profiling, Y_profiling, plaintexts_profiling)
    val_data = (X_val, Y_val, plaintexts_val)

    print(f"✅ Data splits created:")
    print(f"   Training (profiling): {len(train_data[0])} traces")
    print(f"   Validation (attack subset): {len(val_data[0])} traces") 
    print(f"   Test (reserved attack subset): {len(test_indices)} traces")
    print(f"   📋 Validation indices: {len(val_indices)} traces (indices saved for exclusion)")
    print(f"   📋 Test indices: {len(test_indices)} traces (reserved for testing script)")

    return (train_data, val_data), (mean, std), val_indices, test_indices

class ASCADDataset(Dataset):
    def __init__(self, traces, labels, plaintexts, tokenizer):
        self.traces = traces
        self.labels = labels
        self.plaintexts = plaintexts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        label = self.labels[idx]
        plaintext = self.plaintexts[idx]

        # Convert plaintext bytes to hex string
        plaintext_str = ' '.join([f'{b:02x}' for b in plaintext])

        # Tokenize the plaintext
        encoding = self.tokenizer.encode_plus(
            plaintext_str,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'trace': torch.tensor(trace, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    """Custom collate function to stack individual dictionary fields into batched tensors."""
    collated = {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'trace': torch.stack([item['trace'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }
    return collated

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def call(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class BERT_SCA_Model(nn.Module):
    def __init__(self, bert_model, trace_length, ablation=None, dropout_rate=0.2):
        super(BERT_SCA_Model, self).__init__()
        self.ablation = ablation
        self.bert = bert_model

        # Trace encoder config
        trace_dim = 128 if ablation == "tiny_emb" else 768
        self.trace_fc = nn.Linear(trace_length, trace_dim)
        self.trace_bn = nn.BatchNorm1d(trace_dim)

        # Adjust dropout
        drop_rate = 0.0 if ablation == "no_dropout" else dropout_rate
        self.dropout = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)

        # Fusion layer
        if ablation == "no_fusion":
            intermediate_input_dim = trace_dim if ablation == "trace_only" else 768
        else:
            intermediate_input_dim = 768 + trace_dim

        self.intermediate = nn.Linear(intermediate_input_dim, 512)
        self.intermediate_bn = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, 256)

    def forward(self, input_ids, attention_mask, traces):
        # Optional position masking
        position_ids = torch.zeros_like(input_ids) if self.ablation == "no_posenc" else None
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        bert_output = self.dropout(bert_out.pooler_output)

        trace_features = self.trace_fc(traces)
        trace_features = self.trace_bn(trace_features)
        trace_features = self.dropout(trace_features)

        # Handle ablations
        if self.ablation == "no_fusion":
            combined = trace_features
        else:
            combined = torch.cat((bert_output, trace_features), dim=1)

        x = self.intermediate(combined)
        x = self.intermediate_bn(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        return self.classifier(x)

def verify_disjoint_plaintexts_and_keys(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        profiling_metadata = f['Profiling_traces/metadata'][:]
        attack_metadata = f['Attack_traces/metadata'][:]

        profiling_plaintexts = {tuple(pt) for pt in profiling_metadata['plaintext']}
        attack_plaintexts = {tuple(pt) for pt in attack_metadata['plaintext']}
        profiling_keys = {tuple(k) for k in profiling_metadata['key']}
        attack_keys = {tuple(k) for k in attack_metadata['key']}

        overlap_plaintext = profiling_plaintexts & attack_plaintexts
        overlap_key = profiling_keys & attack_keys

        print("🔍 Checking for overlap between profiling and attack sets...")
        if overlap_plaintext:
            print(f"⚠️ Warning: {len(overlap_plaintext)} overlapping plaintexts found!")
        else:
            print("✅ No overlapping plaintexts.")

        if overlap_key:
            print(f"⚠️ Warning: {len(overlap_key)} overlapping keys found!")
        else:
            print("✅ No overlapping keys.")

def evaluate_model(model, data_loader, criterion, device, desc="Evaluation"):
    """Evaluate model on given data loader"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            traces = batch['trace'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, traces)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy

def train_model(args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    dataset_name = os.path.basename(args.dataset_path).split('.')[0]
    project_name = f"bert-sca-{dataset_name}"
    run_name = (
        f"{dataset_name}_e{args.epochs}_bs{args.batch_size}_lr{args.learning_rate:.0e}"
        f"{'_ds' + str(args.downsample) if args.downsample else ''}"
        f"_split-attack_val{args.val_size:.1f}_seed{args.seed}"
    )

    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "dataset_path": args.dataset_path,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "downsample": args.downsample,
            "val_size": args.val_size,
            "split_strategy": "attack_split",
            "embedding": "hex",
            "backbone": "bert-base-uncased",
            "trace_net": "fc",
            "fusion": "concat",
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "early_stopping_patience": args.early_stopping_patience,
            "dropout_rate": args.dropout_rate,
            "use_early_stopping": args.early_stopping_patience > 0
        }
    )

    # Load the dataset with proper attack split and index tracking
    (train_data, val_data), (profiling_mean, profiling_std), val_indices, test_indices = load_ascad_data_with_attack_split(
        args.dataset_path, 
        downsample=args.downsample,
        val_size=args.val_size,
        random_state=args.seed
    )

    # Save normalization stats and indices for testing script
    np.save(output_dir / 'profiling_mean.npy', profiling_mean)
    np.save(output_dir / 'profiling_std.npy', profiling_std)
    np.save(output_dir / 'validation_indices.npy', val_indices) 
    np.save(output_dir / 'test_indices.npy', test_indices)        

    print("✅ Profiling mean/std and attack indices saved for consistent evaluation.")
    print(f"📋 Validation indices saved: {len(val_indices)} traces")
    print(f"📋 Test indices saved: {len(test_indices)} traces")

    # Unpack data
    X_train, Y_train, plaintexts_train = train_data
    X_val, Y_val, plaintexts_val = val_data

    # Randomize labels if requested (for sanity check)
    if args.randomize_labels:
        print("🔀 Randomizing labels for sanity check!")
        np.random.shuffle(Y_train)

    trace_length = X_train.shape[1]

    # Create datasets and DataLoaders 
    train_dataset = ASCADDataset(X_train, Y_train, plaintexts_train, tokenizer)
    val_dataset = ASCADDataset(X_val, Y_val, plaintexts_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn)

    # Initialize model
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BERT_SCA_Model(bert_model, trace_length, ablation=args.ablation, dropout_rate=args.dropout_rate).to(DEVICE)

    # Log ablation to WandB
    wandb.config.update({"ablation": args.ablation})

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )

    # Initialize early stopping
    early_stopping = None
    if args.early_stopping_patience > 0:
        early_stopping = EarlyStopping(patience=args.early_stopping_patience)
        print(f"✅ Early stopping enabled with patience {args.early_stopping_patience}")
    else:
        print("⚠️ Early stopping disabled - training for full epochs")

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            traces = batch['trace'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, traces)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * correct / total

        # Validation phase (only on validation subset of attack data)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, DEVICE, f"Validation Epoch {epoch+1}")

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%), Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%)")

        # Track best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            print(f"📊 New best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

        # Early stopping check
        if early_stopping is not None:
            if early_stopping.call(val_loss, model):
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break

    # ===================================================================
    # 🚫 NO TEST EVALUATION HERE! Test data never seen during training!
    # ===================================================================
    print("\n" + "=",60)
    print("✅ TRAINING COMPLETE - TEST DATA NEVER SEEN!")
    print("=",60)

    # Save final model
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"bertsca{os.path.basename(args.dataset_path).split('.')[0]}_seed{args.seed}_e{args.epochs}_lr{args.learning_rate:.0e}s{args.downsample if args.downsample else 'full'}{timestamp}.pth"    
    model_save_path = output_dir / model_filename
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model saved to {model_save_path}")

    # Save config
    config_filename = model_filename.replace(".pth", "_config.yaml")
    config_save_path = output_dir / config_filename
    config_dict = {
        "dataset_path": args.dataset_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "downsample": args.downsample,
        "val_size": args.val_size,
        "split_strategy": "attack_split",
        "embedding": "hex",
        "backbone": "bert-base-uncased",
        "trace_net": "fc",
        "fusion": "concat",
        "key_byte": 0,
        "num_traces": 10000,
        "random_seed": args.seed,
        "profiling_mean_path": str(output_dir / 'profiling_mean.npy'),
        "profiling_std_path": str(output_dir / 'profiling_std.npy'),
        "validation_indices_path": str(output_dir / 'validation_indices.npy'),
        "test_indices_path": str(output_dir / 'test_indices.npy'),
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "early_stopping_patience": args.early_stopping_patience,
        "dropout_rate": args.dropout_rate,
        "use_early_stopping": args.early_stopping_patience > 0,
        "best_val_loss": best_val_loss,
        "best_val_epoch": best_epoch,
        "val_traces_count": len(val_indices),
        "test_traces_reserved": len(test_indices)
    }

    import yaml
    with open(config_save_path, 'w') as f:
        yaml.dump(config_dict, f)
    print(f"✅ Config saved to {config_save_path}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT SCA model on ASCAD dataset with proper attack split")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the ASCAD dataset (.h5 file)")
    parser.add_argument("--ablation", type=str, default=None, choices=[None, "no_dropout", "no_posenc", "no_fusion", "tiny_emb"], help="Ablation variant to run")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--downsample", type=int, default=None, help="Optional downsampling size of the profiling set")
    parser.add_argument("--val_size", type=float, default=0.3, help="Fraction of attack data to use for validation (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=222, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save models and configs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="Early stopping patience")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate for regularization")
    parser.add_argument("--randomize_labels", action="store_true", help="Randomize labels for sanity check")
    args = parser.parse_args()

    # Set the random seed
    set_seed(args.seed)

    # Verify disjoint plaintexts and keys
    verify_disjoint_plaintexts_and_keys(args.dataset_path)

    train_model(args)