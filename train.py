import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel  # To load HuBERT-ECG
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
import glob
import os
from scipy.io import loadmat
from scipy import signal
import warnings

# --- 1. Configuration ---

# --- IMPORTANT: Change these paths ---
LEFT_DATA_DIR = "/home/gabbru/Desktop/padhy sir/334/left" # 77 files
RIGHT_DATA_DIR = "/home/gabbru/Desktop/padhy sir/334/right" # 257 files
# -------------------------------------

# Pre-trained model we are using
PRETRAINED_MODEL_NAME = "Edoardo-BS/hubert-ecg-small"

# .hea file info
GAIN = 1000.0       # 1000/mV
SAMPLING_RATE = 500.0   # 500 Hz

# Training Params
NUM_CLASSES = 2  # LVOT (0) and RVOT (1)
LEARNING_RATE = 1e-4
BATCH_SIZE = 8   # Use a small batch size for a small dataset
EPOCHS = 30      # We don't need many epochs for fine-tuning
TEST_SPLIT_SIZE = 0.2 # Hold out 20% of data for validation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2. Data Processing (Re-used from pre-training) ---

def build_bandpass_filter(fs, lowcut=0.5, highcut=40.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, b, a):
    return signal.filtfilt(b, a, data, axis=1)

class LabeledECGDataset(Dataset):
    """
    Custom Dataset for our 334 Labeled RVOT/LVOT files.
    It returns (signal, label).
    Handles transposition and cropping.
    """
    def __init__(self, file_paths, labels, gain, fs, mode='train', target_length=5000):
        self.file_paths = file_paths
        self.labels = labels
        self.gain = gain
        self.b, self.a = build_bandpass_filter(fs)
        self.mode = mode # 'train' or 'val'
        self.target_length = target_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # --- STEP 1: Load the correct key ---
            mat_data = loadmat(file_path)
            raw_signal = mat_data['ecgmat'].astype(np.float32) # Use 'ecgmat'

            # --- STEP 2: Transpose the signal ---
            # from (samples, leads) to (leads, samples)
            # (14265, 12) -> (12, 14265)
            if raw_signal.shape[1] == 12:
                raw_signal = raw_signal.T
            
            # Check for correct lead count
            if raw_signal.shape[0] != 12:
                print(f"Warning: Skipping {file_path}. Expected 12 leads, got {raw_signal.shape[0]}")
                return None
            
            # --- STEP 3: Crop the signal ---
            current_length = raw_signal.shape[1]
            if current_length < self.target_length:
                print(f"Warning: Skipping {file_path}. Signal shorter than {self.target_length} samples.")
                return None
            
            if current_length > self.target_length:
                if self.mode == 'train':
                    # Random crop for training
                    start_idx = np.random.randint(0, current_length - self.target_length + 1)
                else:
                    # Center crop for validation
                    start_idx = (current_length - self.target_length) // 2
                
                raw_signal = raw_signal[:, start_idx : start_idx + self.target_length]

            # --- (Rest of the steps are the same) ---

            # 4. Apply Gain
            signal_mv = raw_signal / self.gain

            # 5. Filter Noise
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                filtered_signal = apply_filter(signal_mv, self.b, self.a)

            # 6. Normalize
            mean = np.mean(filtered_signal, axis=1, keepdims=True)
            std = np.std(filtered_signal, axis=1, keepdims=True)
            std[std == 0] = 1.0
            norm_signal = (filtered_signal - mean) / std

            # 7. Format as Tensor
            return torch.from_numpy(norm_signal.copy()).float(), torch.tensor(label, dtype=torch.long)
        
        except KeyError:
            print(f"Error: Key 'ecgmat' not found in {file_path}. Skipping.")
            return None
        except Exception as e:
            print(f"Error loading {file_path}: {e}. Skipping.")
            return None

def collate_fn_skip_corrupt(batch):
    """Custom collate function to filter out None (corrupt) samples."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

# --- 3. Fine-Tuning Model Architecture ---

class HubertForArrhythmiaClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Load the pre-trained HuBERT model
        print("Loading pre-trained HuBERT-ECG model...")
        self.hubert = AutoModel.from_pretrained(
            PRETRAINED_MODEL_NAME, 
            trust_remote_code=True
        )
        
        # 2. Freeze most params, but unfreeze the top 4 layers
        print("Freezing base model and unfreezing top 4 layers...")
        for name, param in self.hubert.named_parameters():
            param.requires_grad = False  # Freeze everything by default
            
            # Unfreeze the top 4 transformer layers (8, 9, 10, 11)
            # This gives the model more flexibility to adapt
            if "transformer.layers.8." in name or \
               "transformer.layers.9." in name or \
               "transformer.layers.10." in name or \
               "transformer.layers.11." in name:
                param.requires_grad = True

        # 3. Add a new, *unfrozen* classification head
        hidden_size = self.hubert.config.hidden_size # (768 for 'small')
        num_leads = 12
        
        # New classifier input size is (num_leads * hidden_size)
        self.classifier = nn.Linear(num_leads * hidden_size, num_classes)
        
        print(f"Model loaded. Top 4 layers and new classifier ({num_leads * hidden_size} -> {num_classes}) are trainable.")

    def forward(self, x):
        # x shape is (Batch, 1, 12, 5000) or (Batch, 12, 5000)
        
        # Ensure we have (Batch, Leads, Length)
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1) # Get rid of potential extra dimension
        
        batch_size, num_leads, seq_len = x.shape
        
        # Reshape to treat each lead as a separate sequence
        # (Batch, Leads, Length) -> (Batch * Leads, Length)
        # Example: (8, 12, 5000) -> (96, 5000)
        x_reshaped = x.reshape(batch_size * num_leads, seq_len)
        
        # Pass the reshaped input through HuBERT
        # Input: (96, 5000)
        # Output: (96, NumPatches, HiddenSize)
        outputs = self.hubert(x_reshaped).last_hidden_state
        
        # --- Aggregate features across leads ---
        num_patches = outputs.shape[1]
        hidden_size = outputs.shape[2]
        
        # Reshape back to separate leads: 
        # (Batch * Leads, NumPatches, HiddenSize) -> (Batch, Leads, NumPatches, HiddenSize)
        outputs_per_lead = outputs.reshape(batch_size, num_leads, num_patches, hidden_size)
        
        # Average features across all patches, but *keep* the leads separate
        # (Batch, Leads, NumPatches, HiddenSize) -> (Batch, Leads, HiddenSize)
        pooled_patches = torch.mean(outputs_per_lead, dim=2) 
        
        # Flatten the lead features into one long vector
        # (Batch, Leads, HiddenSize) -> (Batch, Leads * HiddenSize)
        # e.g., (8, 12, 768) -> (8, 9216)
        pooled_output = pooled_patches.reshape(batch_size, -1)

        # Pass the new long vector through our new classifier
        # (Batch, 9216) -> (Batch, 2)
        logits = self.classifier(pooled_output)
        
        return logits
# --- 4. Training Script ---

def main():
    # 1. Load file paths and create labels
    # Label 0 = LVOT (left), Label 1 = RVOT (right)
    left_files = glob.glob(os.path.join(LEFT_DATA_DIR, "*.mat"))
    left_labels = [0] * len(left_files)
    
    right_files = glob.glob(os.path.join(RIGHT_DATA_DIR, "*.mat"))
    right_labels = [1] * len(right_files)
    
    all_files = left_files + right_files
    all_labels = left_labels + right_labels
    
    print(f"Found {len(left_files)} LVOT (left) samples and {len(right_files)} RVOT (right) samples.")
    print(f"Total: {len(all_files)} labeled samples.")
    
    if not all_files:
        print(f"Error: No files found. Check your LEFT_DATA_DIR and RIGHT_DATA_DIR paths.")
        return

    # 2. Create Train/Validation Split
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels,
        test_size=TEST_SPLIT_SIZE,
        random_state=42,       # For reproducibility
        stratify=all_labels    # Make sure split has same % of left/right
    )
    
    print(f"Training on {len(train_files)} samples. Validating on {len(val_files)} samples.")

    # 3. Create Datasets and DataLoaders
    train_dataset = LabeledECGDataset(train_files, train_labels, GAIN, SAMPLING_RATE, mode='train')
    val_dataset = LabeledECGDataset(val_files, val_labels, GAIN, SAMPLING_RATE, mode='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn_skip_corrupt
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn_skip_corrupt
    )

    # 4. Handle Class Imbalance
    num_lvot = sum(1 for label in train_labels if label == 0)
    num_rvot = len(train_labels) - num_lvot
    
    # Weight = total / (num_classes * num_samples_for_class)
    weight_lvot = len(train_labels) / (2.0 * num_lvot)
    weight_rvot = len(train_labels) / (2.0 * num_rvot)
    
    class_weights = torch.tensor([weight_lvot, weight_rvot]).float().to(DEVICE)
    print(f"Class Weights: LVOT(0)={weight_lvot:.2f}, RVOT(1)={weight_rvot:.2f}")

    # 5. Setup Model, Loss, and Optimizer
    model = HubertForArrhythmiaClassification(num_classes=NUM_CLASSES).to(DEVICE)
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # --- CRITICAL (NEW) ---
    # We pass ALL parameters that are "trainable" to the optimizer
    # This now includes the classifier AND the unfrozen layers
    
    # Use a much smaller learning rate for fine-tuning
    FINETUNE_LR = 5e-5 
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=FINETUNE_LR
    )
    # ----------------
    
    best_val_f1 = 0.0

    print("--- Starting Fine-Tuning ---")

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        
        for signals, labels in train_loader:
            if signals.nelement() == 0: continue # Skip empty batches
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(signals)
            loss = loss_fn(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for signals, labels in val_loader:
                if signals.nelement() == 0: continue
                signals, labels = signals.to(DEVICE), labels.to(DEVICE)
                
                logits = model(signals)
                loss = loss_fn(logits, labels)
                total_val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro') # 'macro' is good for imbalance

        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc*100:.2f}% | Val F1-Score (Macro): {val_f1:.4f}")
        
        # Print confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(f"  Confusion Matrix:\n  {cm}")

        # Save the best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_finetuned_model.pth")
            print(f"  >>> New best model saved with F1-Score: {best_val_f1:.4f}")

if __name__ == "__main__":
    main()