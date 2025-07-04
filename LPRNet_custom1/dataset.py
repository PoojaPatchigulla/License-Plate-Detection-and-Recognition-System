# # 3. dataset.py

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from alphabet import alphabet

class LPRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((48, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.char2idx = {char: idx for idx, char in enumerate(alphabet)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        image = image.resize((96, 48))
        label_str = self.annotations.iloc[idx, 1]
        label_encoded = [self.char2idx[char] for char in label_str]

        if self.transform:
            image = self.transform(image)

        return image, label_encoded
