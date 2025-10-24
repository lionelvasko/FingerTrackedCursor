# helpers/model_handlandmark.py
import torch.nn as nn
import torchvision.models as models

class HandLandmarkNet(nn.Module):
    def __init__(self):
        super().__init__()
        print("   🧠 HandLandmarkNet inicializálása...")
        print("      📦 Backbone: MobileNetV2 (ImageNet súlyokkal)")
        base = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.backbone = base.features
        print("      🔗 FC rétegek: 1280 → 512 → 42 (21 landmark × 2 koordináta)")
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 42),  # 21 * (x, y)
        )
        print("      ✅ Modell architektúra kész")

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x.view(-1, 21, 2)
