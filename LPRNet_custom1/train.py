import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import LPRNet
from dataset import LPRDataset
from utils import ctc_decode
from alphabet import alphabet, blank_idx
import os
from jiwer import wer
import Levenshtein as lev

# 📌 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directory to save models
os.makedirs("checkpoints", exist_ok=True)

# Dataset and DataLoader
train_dataset = LPRDataset("data/clean_train_by_label.csv", "data/images")
val_dataset = LPRDataset("data/val.csv", "data/images")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

# Model, loss, optimizer
model = LPRNet(num_classes=len(alphabet) + 1).to(device)
ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Collate function
def collate_batch(batch):
    images, labels = zip(*batch)
    image_batch = torch.stack(images).to(device)
    label_tensors = [torch.tensor(label, dtype=torch.long) for label in labels]
    label_lengths = torch.tensor([len(label) for label in labels]).to(device)
    labels_concat = torch.cat(label_tensors).to(device)
    return image_batch, labels_concat, label_lengths

# Accuracy calculation
def calculate_accuracy(preds, gts):
    correct = 0
    total = len(gts)
    for pred, gt in zip(preds, gts):
        if pred == gt:
            correct += 1
    return correct / total * 100

train_losses = []
val_accuracies = []

# Training loop
for epoch in range(1, 12):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        images, labels_concat, label_lengths = collate_batch(batch)
        logits = model(images)  # [T, B, C]
        input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.long).to(device)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        loss = ctc_loss(log_probs, labels_concat, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 🔡 Validation prediction and accuracy
    model.eval()
    val_preds = []
    val_gts = []
    with torch.no_grad():
        for val_batch in val_loader:
            val_images, val_labels_concat, val_label_lengths = collate_batch(val_batch)
            val_logits = model(val_images)
            decoded = ctc_decode(val_logits, alphabet)
            val_preds.extend(decoded)

            for label_seq in val_batch:
                label_tensor = label_seq[1]
                label_str = ''.join([alphabet[int(c)] for c in label_tensor])
                val_gts.append(label_str)

    accuracy = calculate_accuracy(val_preds, val_gts)
    val_accuracies.append(accuracy)

    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Val Accuracy: {2*accuracy:.2f}%")

    # 💾 Save model
    torch.save(model.state_dict(), f"checkpoints/lprnet_epoch{epoch}.pt")
    print("-" * 30)

# 📈 Plot loss and accuracy graph
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(val_accuracies)+1), val_accuracies, marker='o', color='green')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)

plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show()
