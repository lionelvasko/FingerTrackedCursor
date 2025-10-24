# train.py
import os
import torch
from torch.utils.data import DataLoader
from dataset_preprocess import preprocess_freihand
from dataset_loader import FreiHANDLandmarkDataset
from model_handlandmark import HandLandmarkNet
from train_handlandmark import train_model
from export_onnx import export_onnx

print("=" * 60)
print("KÉZTARTÁS LANDMARK MODELL TANÍTÁS ELINDÍTVA")
print("=" * 60)

# --- PATHS ---
TRAIN_DIR = r"D:\FreiHAND_pub_v2"
VAL_DIR = r"D:\FreiHAND_pub_v2_eval"

PROCESSED_TRAIN = r"D:\processed\train"
PROCESSED_VAL = r"D:\processed\val"

print(f"\n📁 Beállított útvonalak:")
print(f"   - Tanító adatok: {TRAIN_DIR}")
print(f"   - Validációs adatok: {VAL_DIR}")
print(f"   - Feldolgozott tanító: {PROCESSED_TRAIN}")
print(f"   - Feldolgozott validációs: {PROCESSED_VAL}")

# --- STEP 1: Preprocess ---
print("\n" + "=" * 60)
print("1. LÉPÉS: ADATOK ELŐFELDOLGOZÁSA")
print("=" * 60)

def needs_preprocess(path):
    img_dir = os.path.join(path, "images")
    return not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0

if needs_preprocess(PROCESSED_TRAIN):
    print("🔄 Tanító adatok előfeldolgozása...")
    preprocess_freihand(TRAIN_DIR, PROCESSED_TRAIN)
else:
    print("✅ Tanító adatok már előfeldolgozva, kihagyom.")

if needs_preprocess(PROCESSED_VAL):
    print("🔄 Validációs adatok előfeldolgozása...")
    preprocess_freihand(VAL_DIR, PROCESSED_VAL)
else:
    print("✅ Validációs adatok már előfeldolgozva, kihagyom.")

# --- STEP 2: Load datasets ---
print("\n" + "=" * 60)
print("2. LÉPÉS: ADATHALMAZOK BETÖLTÉSE")
print("=" * 60)
train_ds = FreiHANDLandmarkDataset(PROCESSED_TRAIN)
val_ds = FreiHANDLandmarkDataset(PROCESSED_VAL)
print(f"   ✅ Betöltve: {len(train_ds)} tanító, {len(val_ds)} validációs minta")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

print(f"   📊 Tanító batch-ek: {len(train_loader)}")
print(f"   📊 Validációs batch-ek: {len(val_loader)}")

# --- STEP 3: Model init & training ---
print("\n" + "=" * 60)
print("3. LÉPÉS: MODELL INICIALIZÁLÁS ÉS TANÍTÁS")
print("=" * 60)

device = "cuda"
print(f"🖥️  Használt eszköz: {device.upper()}")

model = HandLandmarkNet().to(device)
print("   ✅ Modell létrehozva")

print("\n🚀 Tanítás elindítása (epochs=4, lr=1e-4)...")
train_model(model, train_loader, val_loader, epochs=4, lr=1e-4, device=device)

# --- STEP 4: Export ---
print("\n" + "=" * 60)
print("4. LÉPÉS: ONNX EXPORT")
print("=" * 60)
export_path = "hand_landmark.onnx"
export_onnx(model, export_path)
print(f"💾 Modell exportálva: {export_path}")

print("\n" + "=" * 60)
print("✅ TANÍTÁSI FOLYAMAT SIKERESEN BEFEJEZVE!")
print("=" * 60)
