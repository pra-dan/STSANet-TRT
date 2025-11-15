import os
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

packet_size = 32
TRT_ENGINE_PATH = 'STSANet_DHF1K.engine'

def image_transform(img):
    # img: HWC, BGR uint8
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 224))  # (width, height)
    img = img.astype(np.float16) / 255.0 # as we have engine exported as fp16
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # C, H, W
    return img

def save_prediction(prediction, save_name, save_size=(384, 224)):
    prediction = cv2.resize(prediction, save_size)
    prediction_norm = (prediction - np.min(prediction)) / (
            (np.max(prediction) - np.min(prediction) + 1e-5) * 1.0)
    prediction_norm = np.round(prediction_norm * 255 + 0.5)
    prediction_norm = np.asarray(prediction_norm, dtype=np.uint8)
    cv2.imwrite(save_name, prediction_norm)

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.allocations = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding))
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.allocations.append(device_mem)
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem, 'binding': binding})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'binding': binding})

    def infer(self, input_array):
        # input_array: (1, 3, packet_size, 224, 384) float32
        np.copyto(self.inputs[0]['host'], input_array.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        infer_start = time.time()
        self.context.execute_v2([int(a) for a in self.allocations])
        infer_end = time.time()
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
        # Output shape: (1, 1, H, W) or (1, H, W)
        output_shape = self.engine.get_tensor_shape(self.outputs[0]['binding'])
        output = np.array(self.outputs[0]['host']).reshape(output_shape)
        if output.ndim == 4:
            output = output[0, 0]
        elif output.ndim == 3:
            output = output[0]
        return output, infer_end-infer_start

def video_inference_trt(video_file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- START DEBUGGING ---
    if not os.path.exists(video_file_path):
        print(f"ERROR: Video file not found at: {video_file_path}")
        return
    print(f"File exists at: {video_file_path}")

    cap = cv2.VideoCapture(video_file_path)

    if not cap.isOpened():
        print("\nERROR: cap.isOpened() is False.")
        print("cv2.VideoCapture() failed to open the video file.")
        print("This is almost certainly because OpenCV is missing the FFmpeg backend.\n")
        
        # This will print a massive log. Look for the 'Video I/O' section.
        print("--- OpenCV Build Information ---")
        print(cv2.getBuildInformation())
        print("---------------------------------")
        print("In the 'Video I/O' section above, you will likely see 'FFMPEG: NO'")
        return
    # --- END DEBUGGING ---

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));print("total frames ",frame_count)

    trt_infer = TRTInference(TRT_ENGINE_PATH)

    transformed_frames = []
    frame_index = 0
    saliency_stack = []

    total_infer_time = 0.0
    infer_count = 0

    start_time = time.time()

    # import pdb;pdb.set_trace()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("safely breaking from stream")
            break

        # Preprocess frame
        frame_tensor = image_transform(frame)
        if len(transformed_frames) == packet_size:
            transformed_frames.pop(0)
        transformed_frames.append(frame_tensor)

        # Only predict if we have enough frames
        if len(transformed_frames) == packet_size:
            # Stack frames: (packet_size, 3, 224, 384) -> (1, 3, packet_size, 224, 384)
            clip = np.stack(transformed_frames, axis=0)  # (packet_size, 3, 224, 384)
            clip = np.transpose(clip, (1, 0, 2, 3))      # (3, packet_size, 224, 384)
            clip = np.expand_dims(clip, axis=0).astype(np.float16)  # (1, 3, packet_size, 224, 384)
            
            prediction, infer_time = trt_infer.infer(clip)
            
            total_infer_time += infer_time
            infer_count += 1
            # Save prediction for this frame
            save_name = os.path.join(output_dir, f"frame_{frame_index:05d}_saliency.png")
            save_prediction(prediction, save_name, save_size=(width, height))
            saliency_stack.append(prediction)
        frame_index += 1
        print(f"Processed frame {frame_index}/{frame_count}")

    cap.release()
    end_time = time.time()
    total_time = end_time - start_time
    avg_infer_time = total_infer_time / infer_count if infer_count > 0 else 0.0
    print(f"Processed {frame_index} frames in {total_time:.2f} seconds.")
    print(f"Total inference calls: {infer_count}")
    print(f"Average inference time per frame: {avg_infer_time*1000:.2f} ms")
    print(f"Total inference time: {total_infer_time:.2f} seconds.")

    # Save overlayed video and saliency video (like infer_onnx.py)
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
        # Normalize and resize saliency map
        sal_norm = (sal - np.min(sal)) / (np.max(sal) - np.min(sal) + 1e-5)
        sal_frame = cv2.resize((sal_norm * 255).astype(np.uint8), (width, height))
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
    output_dir = "./output/trt"
    video_inference_trt(video_file_path, output_dir)