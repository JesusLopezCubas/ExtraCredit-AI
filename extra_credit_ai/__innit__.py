from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# Path to your folder in Google Drive (adjust if needed)
base_path = "/content/drive/MyDrive/audio_data"

# Load CSV files from Drive
train_df = pd.read_csv(f"{base_path}/train.csv")
test_df = pd.read_csv(f"{base_path}/test.csv")
label_map = pd.read_csv(f"{base_path}/_label_map.csv")
sample_submission = pd.read_csv(f"{base_path}/sample_submission.csv")

# Print shapes
print("✅ Train shape:", train_df.shape)
print("✅ Test shape:", test_df.shape)
print("✅ Label map shape:", label_map.shape)

#%%
import os
import io
import gzip
import base64
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import torchaudio
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# ========== ENV SETUP ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ========== CONFIG ==========
# Training hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 25  # ⬆️ Longer training for deeper generalization

# Audio settings
ORIG_SR = 16000            # Original sample rate
TARGET_SR = 8000           # Downsampled sample rate
MAX_LEN = TARGET_SR * 10   # 10 seconds → 80000 samples

# Mel spectrogram settings
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

# System
NUM_WORKERS = 2  # Adjustable based on environment

# Paths
CHECKPOINT = "rawcnn_baseline.pt"
SUBMISSION = "submission.csv"

# Augmentation
MIXUP_ALPHA = None  # Placeholder if we add mixup later

#%%
# Augmentation class
class WaveformAugment:
    def __init__(self, noise_level=0.005, shift_ms=100, sample_rate=TARGET_SR):
        self.noise_level = noise_level
        self.shift = int(shift_ms / 1000 * sample_rate)

    def __call__(self, waveform):
        gain = torch.rand(1, device=waveform.device) * (1.5 - 0.5) + 0.5
        waveform = waveform * gain
        noise = torch.randn_like(waveform) * self.noise_level
        waveform = waveform + noise
        offset = torch.randint(-self.shift, self.shift + 1, (1,)).item()
        waveform = torch.roll(waveform, shifts=offset)
        return waveform


# Resampler and MelSpectrogram
resampler = torchaudio.transforms.Resample(orig_freq=ORIG_SR, new_freq=TARGET_SR)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

# Decoding and preprocessing function
def decode_audio(b64_string, compressed=True):
    raw = base64.b64decode(b64_string)
    if compressed:
        raw = gzip.decompress(raw)
    waveform, sr = torchaudio.load(io.BytesIO(raw))
    waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = resampler(waveform)
    return waveform.squeeze(0)


# Dataset class — now returns mel spectrograms
class MelSpectrogramDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.aug = WaveformAugment()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform = decode_audio(row['encoded_audio'])

        # Pad or trim
        if waveform.size(0) > MAX_LEN:
            waveform = waveform[:MAX_LEN]
        else:
            waveform = F.pad(waveform, (0, MAX_LEN - waveform.size(0)))

        if self.augment:
            waveform = self.aug(waveform)

        # Normalize waveform before Mel conversion
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)

        # Convert to Mel spectrogram
        mel_spec = mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-6)  # Log scale for stability

        return mel_spec.unsqueeze(0), row['class']

#%%
class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class MelResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = ResidualBlock2D(16, 32, stride=2)
        self.layer2 = ResidualBlock2D(32, 64, stride=2)
        self.layer3 = ResidualBlock2D(64, 128, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)         # (B, 1, 64, T) → (B, 16, H/2, W/2)
        x = self.layer1(x)       # Downsample spatially
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)  # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

#%%
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import multiprocessing
import numpy as np

# Path to your folder in Google Drive
base_path = "/content/drive/MyDrive/audio_data"

# Load and encode labels
df = pd.read_csv(f"{base_path}/train.csv")
le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])  # Convert to integer class labels

# Number of classes
num_classes = len(le.classes_)

# Stratified train/val split
train_df, val_df = train_test_split(df, stratify=df['class'], test_size=0.2, random_state=42)
print(f"✅ Data sizes — train: {len(train_df)} | val: {len(val_df)}")

# Compute class weights to handle imbalance
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_df['class']), y=train_df['class'])
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Weighted sampling per instance
sample_weights = class_weights[train_df['class'].values]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# ⚠️ Use MelSpectrogramDataset now
train_ds = MelSpectrogramDataset(train_df, augment=True)
val_ds = MelSpectrogramDataset(val_df, augment=False)

# CPU handling
num_workers = min(4, multiprocessing.cpu_count())

# DataLoaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

#%%
import IPython.display as ipd

for i in range(2):
    row = df.iloc[i]
    waveform = decode_audio(row["encoded_audio"])

    print(f"🔊 row_id: {row['row_id']}, class (encoded): {row['class']}")
    print(f"📈 waveform shape: {waveform.shape}, duration: {waveform.shape[0] / ORIG_SR:.2f} seconds")

    ipd.display(ipd.Audio(waveform.numpy(), rate=ORIG_SR))

#%%
# Initialize MelResNet model
model = MelResNet(num_classes=num_classes).to(DEVICE)

# Weighted cross-entropy loss to balance classes
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

# Optimizer and LR scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Tracking metrics
train_losses, val_losses, train_f1s, val_f1s = [], [], [], []
best_f1 = 0.0
patience, patience_counter = 6, 0  # early stopping with patience

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    y_true_train, y_pred_train = [], []

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        y_pred_train.extend(outputs.argmax(1).detach().cpu().numpy())
        y_true_train.extend(labels.cpu().numpy())

    train_loss = running_loss / len(train_loader)
    train_f1 = f1_score(y_true_train, y_pred_train, average='macro')
    train_losses.append(train_loss)
    train_f1s.append(train_f1)

    # Validation
    model.eval()
    running_loss = 0.0
    y_true_val, y_pred_val = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            y_pred_val.extend(outputs.argmax(1).cpu().numpy())
            y_true_val.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    val_f1 = f1_score(y_true_val, y_pred_val, average='macro')
    val_losses.append(val_loss)
    val_f1s.append(val_f1)

    scheduler.step(val_f1)

    print(f"Epoch {epoch}: 🧠 TrainLoss={train_loss:.4f} | TrainF1={train_f1:.4f} | ValLoss={val_loss:.4f} | ValF1={val_f1:.4f}")

    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), CHECKPOINT)
        print("✅ Best model saved.")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

# Plot loss curves
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.tight_layout()
plt.show()

# Plot F1 score curves
plt.figure(figsize=(8, 4))
plt.plot(train_f1s, label='Train F1')
plt.plot(val_f1s, label='Val F1')
plt.xlabel('Epoch')
plt.ylabel('Macro F1')
plt.title('F1 Score Curves')
plt.legend()
plt.tight_layout()
plt.show()

#%%
import pandas as pd

# ==================== INFERENCE CELL ====================

base_path = "/content/drive/MyDrive/audio_data"
SUBMISSION = f"{base_path}/submission_v2.csv"


# 2️⃣ Load the test set from Google Drive
test_df = pd.read_csv(f"{base_path}/test.csv")
test_df['class'] = 0  # Dummy placeholder so we can reuse the dataset class

# 3️⃣ Use the MelSpectrogram dataset
test_ds = MelSpectrogramDataset(test_df, augment=False)
test_loader = DataLoader(test_ds, batch_size=32, num_workers=2, pin_memory=True)

# 4️⃣ Load your trained model
model = MelResNet(num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# 5️⃣ Run predictions
preds = []
with torch.no_grad():
    for waveforms, _ in tqdm(test_loader, desc='Inference'):
        waveforms = waveforms.to(DEVICE)
        outputs = model(waveforms)
        preds.extend(outputs.argmax(dim=1).cpu().numpy())

# 6️⃣ Save submission CSV to Google Drive
submission_df = pd.DataFrame({
    'row_id': test_df['row_id'],
    'class': preds
})
submission_df.to_csv(SUBMISSION, index=False)
print(f"✅ Submission saved to {SUBMISSION}")
