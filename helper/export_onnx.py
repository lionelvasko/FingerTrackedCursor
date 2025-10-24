import torch

def export_onnx(model, export_path):
    print("   📊 ONNX export paraméterek:")
    print(f"      - Fájlnév: {export_path}")
    print("      - Input shape: (1, 3, 224, 224)")
    print("      - Dynamic axes: batch dimension")
    print("   🔧 Dummy input létrehozása...")
    
    dummy_input = torch.randn(1, 3, 224, 224).to("cuda")
    model = model.to("cuda")
    
    print("   📥 Modell eval módba állítása...")
    model.eval()

    print("   💾 ONNX export folyamatban...")
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("   ✅ Export sikeres:", export_path)
