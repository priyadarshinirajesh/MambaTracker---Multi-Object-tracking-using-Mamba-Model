# mamba_feature_extractor.py
import torch
from torchvision import transforms
from PIL import Image
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m").to(device)

def extract_feature(cropped_img):
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    img_tensor = preprocess(cropped_img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img_tensor)
    return features.cpu().numpy()
