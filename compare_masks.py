import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

# --- Configuration ---
# ** Ground Truth / Reference ** (Python torch output)
py_dir = "output/pth"
# ** Prediction ** 
cpp_dir = "output/trt"
# ---------------------

# def compute_nss(pred_map, gt_map):
#     """
#     Computes Normalized Scanpath Saliency (NSS).
#     """
#     pred_map_norm = pred_map.astype(np.float32)
#     mean = np.mean(pred_map_norm)
#     std = np.std(pred_map_norm)
    
#     if std == 0:
#         return 0.0
#     pred_map_norm = (pred_map_norm - mean) / std

#     gt_map_norm = gt_map.astype(np.float32)
#     fixation_mask = gt_map_norm > np.mean(gt_map_norm)
    
#     n_fixations = np.sum(fixation_mask)
#     if n_fixations == 0:
#         return 0.0

#     nss_values = pred_map_norm[fixation_mask]
#     return np.mean(nss_values)

def compute_focal_point_offset(map1, map2):
    """
    Computes the Euclidean distance between the brightest pixels of two maps.
    """
    _, _, _, max_loc1 = cv2.minMaxLoc(map1)
    _, _, _, max_loc2 = cv2.minMaxLoc(map2)
    
    # max_loc is (x, y)
    p1 = np.array(max_loc1)
    p2 = np.array(max_loc2)
    
    # Calculate Euclidean distance
    distance = np.linalg.norm(p1 - p2)
    return distance

# Step 1: Scan python directory for saliency frames
t_frames = []
file_pattern = re.compile(r"frame_(\d{5})_saliency\.png")

print(f"Scanning for frames in: {py_dir}")
try:
    for filename in os.listdir(py_dir):
        match = file_pattern.match(filename)
        if match:
            frame_num = int(match.group(1))
            t_frames.append(frame_num)
except FileNotFoundError:
    print(f"Error: Directory not found: {py_dir}")
    print("Please check the 'py_dir' variable.")
    exit(1)

t_frames.sort()
print(f"Total saliency frames found in directory: {len(t_frames)}")

# Step 2: Compute metrics between corresponding cpp and python outputs
metric_info = []

print("Comparing C++ and Python outputs...")
for f in tqdm(t_frames):
    cpp_path = os.path.join(cpp_dir, f"frame_{f:05d}_saliency.png")
    py_path = os.path.join(py_dir, f"frame_{f:05d}_saliency.png")

    if os.path.exists(cpp_path) and os.path.exists(py_path):
        cpp_img = cv2.imread(cpp_path, cv2.IMREAD_GRAYSCALE)
        py_img_orig = cv2.imread(py_path, cv2.IMREAD_GRAYSCALE)

        if cpp_img is None or py_img_orig is None:
            print(f"Failed to read one of the images at frame {f}")
            continue

        # Resize Python image to match C++ image shape
        # This is CRITICAL for a fair comparison of coordinates and MAE
        if cpp_img.shape != py_img_orig.shape:
            py_img = cv2.resize(py_img_orig, (cpp_img.shape[1], cpp_img.shape[0]))
        else:
            py_img = py_img_orig

        # Compute MAE
        mae = np.mean(np.abs(cpp_img.astype(np.float32) - py_img.astype(np.float32)))
        
        # Compute NSS (C++ as pred, Py as GT)
        # nss = compute_nss(cpp_img, py_img)
        
        # Compute Focal Point Offset
        offset = compute_focal_point_offset(cpp_img, py_img)
        
        metric_info.append((f, mae, offset, cpp_path, py_path))
    else:
        if not os.path.exists(cpp_path):
            print(f"Missing C++ frame for Python frame {f} (looked for {cpp_path})")

print("Finished computing metrics.")

# Step 3: Reporting
if metric_info:
    metric_info.sort(key=lambda x: x[0])  # sort by frame number
    frames, maes, offsets, _, _ = zip(*[
        (f, m, o, c, p) for f, m, o, c, p in metric_info
    ])

    # --- MAE Statistics ---
    print("\n--- MAE Analysis (Lower is Better) ---")
    print(f"Matched frames: {len(metric_info)}")
    print(f"Mean MAE: {np.mean(maes):.2f}")
    print(f"Std MAE:  {np.std(maes):.2f}")
    print(f"Min MAE:  {np.min(maes):.2f}")
    print(f"Max MAE:  {np.max(maes):.2f}")

    # # --- NSS Statistics ---
    # print("\n--- NSS Analysis (Higher is Better) ---")
    # print(f"Mean NSS: {np.mean(nsses):.2f}")
    # print(f"Std NSS:  {np.std(nsses):.2f}")
    # print(f"Min NSS:  {np.min(nsses):.2f}")
    # print(f"Max NSS:  {np.max(nsses):.2f}")

    # --- Focal Point Offset Statistics ---
    print("\n--- Focal Point Offset Analysis (Lower is Better) ---")
    print(f"Mean Offset (pixels): {np.mean(offsets):.2f}")
    print(f"Std Offset (pixels):  {np.std(offsets):.2f}")
    print(f"Min Offset (pixels):  {np.min(offsets):.2f}")
    print(f"Max Offset (pixels):  {np.max(offsets):.2f}")
    
    # Report top N worst offsets
    top_n = 5
    basis_metric_id = 1 # using offset to find top 5 
    offset_info_sorted = sorted(metric_info, key=lambda x: x[basis_metric_id], reverse=True)
    print(f"\nTop {top_n} frames with highest Offset (worst):")
    for f, mae, offset, cpath, ppath in offset_info_sorted[:top_n]:
        print(f"  Frame {f}: Offset={offset:.2f} pixels")


    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), sharex=True)

    # MAE Plot
    ax1.plot(frames, maes, marker='o', markersize=2, linestyle='-', color='blue', label='MAE')
    ax1.set_title("MAE (C++ vs Py) - Lower is Better")
    ax1.set_ylabel("Mean Absolute Error (MAE)")
    ax1.grid(True)
    ax1.legend()

    # NSS Plot
    # ax2.plot(frames, nsses, marker='o', markersize=2, linestyle='-', color='green', label='NSS')
    # ax2.set_title("NSS (C++ as pred, Py as GT) - Higher is Better")
    # ax2.set_ylabel("Normalized Scanpath Saliency (NSS)")
    # ax2.grid(True)
    # ax2.legend()
    
    # Focal Point Offset Plot
    ax2.plot(frames, offsets, marker='o', markersize=2, linestyle='-', color='red', label='Offset')
    ax2.set_title("Focal Point Offset - Lower is Better")
    ax2.set_xlabel("Frame number")
    ax2.set_ylabel("Euclidean Distance (pixels)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plot_path = "all_metrics_plot.png"
    plt.savefig(plot_path)
    print(f"\nMetrics plot saved to {plot_path}")
    plt.show()

else:
    print("No valid frame pairs found for comparison.")