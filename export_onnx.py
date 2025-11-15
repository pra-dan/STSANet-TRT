import torch
from STSA_model import STSANet

def load_weights(model, weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    new_checkpoint = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            k = k[7:]
        new_checkpoint[k] = v
    model.load_state_dict(new_checkpoint, strict=False)

ww = "STSANet_DHF1K.pth"

def export_onnx(onnx_path="STSANet_DHF1K.onnx", weights_path=ww):
    # Load model
    model = STSANet()
    load_weights(model, weights_path, 'cpu')
    model.eval()

    # Example input: (batch, channels, frames, height, width)
    # In video_inference.py, input is (1, 3, 32, 224, 384)
    dummy_input = torch.randn(1, 3, 32, 224, 384)

    with torch.no_grad():
        print(model(dummy_input).shape)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=16,  # or higher if required
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    print(f"Exported ONNX model to {onnx_path}")

if __name__ == "__main__":
    export_onnx()
