import os
import json
import numpy as np
import cv2

def preprocess_freihand(data_dir, output_dir):
    print(f"   🔍 Adatforrás: {data_dir}")
    print(f"   📁 Kimeneti mappa: {output_dir}")

    rgb_dir = os.path.join(data_dir, "training", "rgb")
    xyz_path = os.path.join(data_dir, "training_xyz.json")
    k_path = os.path.join(data_dir, "training_K.json")

    if not os.path.exists(rgb_dir):
        print(f"   ⚠️  RGB mappa nem található: {rgb_dir}")
        return
    if not os.path.exists(xyz_path):
        print(f"   ⚠️  XYZ fájl nem található: {xyz_path}")
        return
    if not os.path.exists(k_path):
        print(f"   ⚠️  Kamera mátrix (K) nem található: {k_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    label_dir = os.path.join(output_dir, "labels")
    img_out_dir = os.path.join(output_dir, "images")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(img_out_dir, exist_ok=True)

    # --- Betöltés ---
    with open(xyz_path, "r") as f:
        xyz_list = json.load(f)
    with open(k_path, "r") as f:
        K_list = json.load(f)
    print(f"   📊 Landmark minták száma: {len(xyz_list)}")

    processed_count, skipped_count = 0, 0

    print(f"   🔄 Képek feldolgozása ({len(xyz_list)} elem)...")
    for idx, pts3d in enumerate(xyz_list):
        img_path = os.path.join(rgb_dir, f"{idx:08d}.jpg")
        if not os.path.exists(img_path):
            skipped_count += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped_count += 1
            continue

        K = np.array(K_list[idx])  # 3x3
        pts3d = np.array(pts3d)    # (21, 3)

        # ✅ 3D -> 2D vetítés homogén koordináta nélkül
        proj = (K @ pts3d.T).T  # (21, 3)
        pts2d = proj[:, :2] / proj[:, 2, np.newaxis]  # normalize by z

        h, w = img.shape[:2]
        pts2d[:, 0] /= w
        pts2d[:, 1] /= h

        # Mentés
        with open(os.path.join(label_dir, f"{idx:08d}.json"), "w") as f:
            json.dump(pts2d.tolist(), f)
        cv2.imwrite(os.path.join(img_out_dir, f"{idx:08d}.jpg"), img)
        processed_count += 1

        if idx % 1000 == 0 and idx > 0:
            print(f"      ⏳ {idx}/{len(xyz_list)} ({processed_count} kész, {skipped_count} kihagyva)")

    print(f"   ✅ Előfeldolgozás kész!")
    print(f"      📊 Feldolgozva: {processed_count} kép")
    print(f"      ⚠️  Kihagyva: {skipped_count} kép")
    print(f"      💾 Kimenet: {output_dir}")
