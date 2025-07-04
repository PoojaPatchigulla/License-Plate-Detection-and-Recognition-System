# lprnet_custom1/lpr_infer.py

import torch
from PIL import Image
from torchvision import transforms
from .model import LPRNet
from .utils import ctc_decode
from .alphabet import alphabet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LPRNet(num_classes=len(alphabet) + 1).to(device)
model.load_state_dict(torch.load("/Users/poojapatchigulla/Downloads/myapp4/LPRNet_custom1/checkpoints/lprnet_epoch116.pt", map_location=device))

model.eval()

transform = transforms.Compose([
    transforms.Resize((24, 94)),
    transforms.ToTensor()
])

def recognize_plate(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        pred_str = ctc_decode(logits, alphabet)[0]
    return pred_str
