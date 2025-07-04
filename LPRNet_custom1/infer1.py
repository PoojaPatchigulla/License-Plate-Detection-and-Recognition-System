# 6. infer.py

import torch
from PIL import Image
from torchvision import transforms
from model import LPRNet
from utils import ctc_decode
from alphabet import alphabet
import pandas as pd
import os

# Config
val_csv = "data/val.csv"
img_dir = "data/images"
model_path = "checkpoints/lprnet_epoch116.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = LPRNet(num_classes=len(alphabet) + 1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Preprocessing (resize to 24x94 and normalize)
transform = transforms.Compose([
    transforms.Resize((24, 94)),
    transforms.ToTensor()
])

# Load validation data
val_data = pd.read_csv(val_csv)

# Loop over each sample in val.csv
for idx, row in val_data.iterrows():
    filename, gt_str = row['filename'], row['label']
    image_path = os.path.join(img_dir, filename)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Shape: [1, 3, 24, 94]

    # Inference
    with torch.no_grad():
        logits = model(image_tensor)
        pred_str = ctc_decode(logits, alphabet)[0]

    print(f"GT: {gt_str} | Prediction: {pred_str}")
