# helpers/train_handlandmark.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-4, device="cuda"):
    print(f"\n🔧 Tanítási paraméterek:")
    print(f"   - Epoch-ok száma: {epochs}")
    print(f"   - Tanulási ráta: {lr}")
    print(f"   - Eszköz: {device}")
    print(f"   - Veszteségfüggvény: MSE Loss")
    print(f"   - Optimalizáló: Adam")
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n📤 Modell áthelyezése eszközre: {device}")
    model.to(device)
    print(f"   ✅ Modell készen áll a tanításra")

    print(f"\n{'='*60}")
    print("TANÍTÁS ELKEZDVE")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        print(f"\n📊 EPOCH {epoch+1}/{epochs}")
        print("-" * 40)
        
        # TRAIN
        model.train()
        train_loss = 0.0
        print(f"🏋️  Tanítás...")
        for imgs, landmarks in tqdm(train_loader, desc=f"  └─ Epoch {epoch+1}/{epochs} [train]"):
            imgs, landmarks = imgs.to(device), landmarks.to(device)
            preds = model(imgs)
            loss = criterion(preds, landmarks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        print(f"🔍 Validáció...")
        with torch.no_grad():
            for imgs, landmarks in tqdm(val_loader, desc=f"  └─ Epoch {epoch+1}/{epochs} [val]"):
                imgs, landmarks = imgs.to(device), landmarks.to(device)
                preds = model(imgs)
                loss = criterion(preds, landmarks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\n📈 EPOCH {epoch+1} EREDMÉNYEK:")
        print(f"   🏋️  Train Loss: {avg_train_loss:.5f}")
        print(f"   🔍 Val Loss:   {avg_val_loss:.5f}")
        print("-" * 40)

    print(f"\n{'='*60}")
    print("✅ TANÍTÁS BEFEJEZVE!")
    print(f"{'='*60}")
