# helpers/dataset_loader.py
import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FreiHANDLandmarkDataset(Dataset):
    def __init__(self, root_dir):
        print(f"   📂 Dataset inicializálása: {root_dir}")
        self.img_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Képek mappája nem található: {self.img_dir}")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Címkék mappája nem található: {self.label_dir}")
        
        self.img_files = sorted(os.listdir(self.img_dir))
        print(f"   📊 Talált képek száma: {len(self.img_files)}")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        print(f"   🔧 Transformációk beállítva: Resize(224x224) + ToTensor()")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".json"))

        image = Image.open(img_path).convert("RGB")
        with open(label_path, "r") as f:
            landmarks = json.load(f)

        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        image = self.transform(image)

        return image, landmarks
