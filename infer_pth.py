import os
import sys
import time
import cv2
import numpy as np
import torch
from torchvision import transforms

import STSA_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

packet_size = 32
STSA_MODEL_PATH = "STSANet_DHF1K.pth"

# Load model
model = STSA_model.STSANet()
# Load weights correctly, handling possible DataParallel or key prefix issues
checkpoint = torch.load(STSA_MODEL_PATH, map_location=device)
if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']
# Remove possible 'module.' prefix
new_checkpoint = {}
for k, v in checkpoint.items():
    if k.startswith('module.'):
        k = k[7:]
    new_checkpoint[k] = v
model.load_state_dict(new_checkpoint, strict=False)
model.to(device)
model.eval()

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def save_prediction(prediction, save_name, save_size=(384, 224)):
    prediction = prediction.cpu().squeeze().clone().numpy()
    prediction = cv2.resize(prediction, save_size)
    prediction_norm = (prediction - np.min(prediction)) / (
            (np.max(prediction) - np.min(prediction) + 0.00001) * 1.0)
    prediction_norm = np.round(prediction_norm * 255 + 0.5)
    prediction_norm = np.asarray(prediction_norm, dtype=np.uint8)
    cv2.imwrite(save_name, prediction_norm)

def video_inference(video_file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    transformed_frames = []
    frame_index = 0
    saliency_stack = []

    start_time = time.time()
    total_pred_time = 0.0
    pred_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame_tensor = image_transform(frame)
        if len(transformed_frames) == packet_size:
            transformed_frames.pop(0)
        transformed_frames.append(frame_tensor)

        # Only predict if we have enough frames
        if len(transformed_frames) == packet_size:
            # Stack frames: (packet_size, 3, 224, 384) -> (1, 3, packet_size, 224, 384)
            clip = torch.stack(transformed_frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(device)
            pred_start = time.time()
            with torch.no_grad():
                prediction = model(clip)
            pred_end = time.time()
            print("single pred time ",1000*(pred_end - pred_start))
            total_pred_time += (pred_end - pred_start)
            pred_frame_count += 1
            # Save prediction for this frame
            save_name = os.path.join(output_dir, f"frame_{frame_index:05d}_saliency.png")
            save_prediction(prediction, save_name, save_size=(width, height))
            saliency_stack.append(prediction.cpu().squeeze().clone().numpy())
        frame_index += 1
        print(f"Processed frame {frame_index}/{frame_count}")

    cap.release()
    end_time = time.time()
    total_time = end_time - start_time
    avg_pred_time = total_pred_time / pred_frame_count if pred_frame_count > 0 else 0.0
    print(f"Processed {frame_index} frames in {total_time:.2f} seconds.")
    print(f"Total inference calls: {pred_frame_count}")
    print(f"Average inference time per frame: {avg_pred_time*1000:.2f} ms")
    print(f"Total inference time: {total_pred_time:.2f} seconds.")

    # Optionally, create overlayed video and saliency video
    overlayed_video_path = os.path.join(output_dir, "overlayed.mp4")
    saliency_video_path = os.path.join(output_dir, "saliency.mp4")
    overlayed_out = cv2.VideoWriter(
        overlayed_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height)
    )
    saliency_out = cv2.VideoWriter(
        saliency_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height), isColor=False
    )

    # Re-read video for overlay
    cap = cv2.VideoCapture(video_file_path)
    counter = 0
    while cap.isOpened() and counter < len(saliency_stack):
        ret, image = cap.read()
        if not ret:
            break
        sal = saliency_stack[counter]
        sal_frame = cv2.resize((sal - np.min(sal)) / (np.max(sal) - np.min(sal) + 1e-5) * 255, (width, height)).astype(np.uint8)
        color_map = cv2.applyColorMap(sal_frame, cv2.COLORMAP_JET)
        overlayed_frame = cv2.addWeighted(image, 0.5, color_map, 0.5, 0)
        overlayed_out.write(overlayed_frame)
        saliency_out.write(sal_frame)
        counter += 1

    cap.release()
    overlayed_out.release()
    saliency_out.release()
    print(f"Overlayed and saliency videos saved to {output_dir}")

if __name__ == "__main__":
    video_file_path = "25s.mp4"
    output_dir = "output/pth"
    video_inference(video_file_path, output_dir)