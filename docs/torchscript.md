# Using the TorchScript VideoSeal model

This document provides instructions on how to use the TorchScript version of the VideoSeal model for watermarking images and videos.

## Loading and using

### Download the model

You can download the TorchScript model with:
```bash
# For Linux/Windows:
wget https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit -P ckpts/

# For Mac:
mkdir ckpts
curl -o ckpts/y_256b_img.jit https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit
```
Or by clicking [here](https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit).

### Image watermarking

```python
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

# Load the JIT model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("ckpts/y_256b_img.jit")
model.to(device)
model.eval()

# Load image.
img_path = "path/to/your/image.jpg"
img = Image.open(img_path).convert("RGB")
img_o = to_tensor(img).unsqueeze(0).float().to(device)

# Create a message to embed (random binary vector of 256bits).
msg = torch.randint(0, 2, (1, 256)).float().to(device)

# Option 1: Combined embedding and detection.
with torch.no_grad():
    # Returns watermarked image and predictions.
    img_w, preds = model(img_o, msg)

# Option 2: Embedding only.
with torch.no_grad():
    # Returns watermarked image directly.
    img_w = model.embed(img_o, msg)

# Convert back to PIL Image for saving.
img_w_pil = to_pil_image(img_w.squeeze().cpu())
save_path = img_path.split(".")[0] + "_wm.jpg"
img_w_pil.save(save_path)

# Option 3: Detection only.
img_w = Image.open(save_path).convert("RGB")
img_w = to_tensor(img_w).unsqueeze(0).float().to(device)
with torch.no_grad():
    # Returns predictions tensor directly.
    preds = model.detect(img_w)
    
    # Process predictions to get binary message.
    # Assuming first channel is detection mask and rest are bit predictions.
    bit_preds = preds[:, 1:]  # Exclude mask
    detected_message = (bit_preds > 0).float()  # Threshold
```

### Video watermarking

```python
import torch
import os
from torchvision.io import read_video, write_video
from torchvision.transforms.functional import to_pil_image

# Load the model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("ckpts/y_256b_img.jit")
model.to(device)
model.eval()

# Load video using torchvision.
video_path = "path/to/your/video.mp4"
# Read video returns a tuple of (video_frames, audio_frames, metadata)
frames, audio, metadata = read_video(video_path, pts_unit='sec')
# Convert to float and normalize to [0, 1].
video_tensor = frames.float() / 255.0
# Move to device and ensure format [T, C, H, W].
video_tensor = video_tensor.permute(0, 3, 1, 2).to(device)

# Create a message to embed (must be of shape 1xK for video).
message = torch.randint(0, 2, (1, 256)).float().to(device)

# Embed watermark in video.
with torch.no_grad():
    # Returns watermarked video directly.
    video_tensor_w = model.embed(video_tensor, message, is_video=True)

# Convert back to uint8 for saving.
watermarked_video = (video_tensor_w.cpu() * 255.0).to(torch.uint8)
# Convert back to format expected by write_video [T, H, W, C].
watermarked_video = watermarked_video.permute(0, 2, 3, 1)

# Save watermarked video with original audio.
try:
    output_path = "watermarked_video.mp4"
    # Get original video fps from metadata.
    fps = metadata["video_fps"]
    write_video(output_path, watermarked_video, fps=fps)
    print(f"Saved watermarked video to {output_path}")
except Exception as e:
    print(e)

# Detect message from watermarked video.
with torch.no_grad():
    # Returns predictions for each frame.
    frame_preds = model.detect(video_tensor_w, is_video=True)
    
    # Aggregate predictions across frames.
    aggregated_msg = model.detect_video_and_aggregate(
        video_tensor_w,
        aggregation="avg"  # Options: "avg", "squared_avg", "l1norm_avg", "l2norm_avg"
    )
    
    # Compare with original message.
    correct = (aggregated_msg == message).float().mean().item()
    print(f"Message recovery accuracy: {correct*100:.2f}%")
```

## Available functions

The TorchScript VideoSeal model provides the following functions:

### Main functions

1. **forward(imgs, msgs, is_video=False)**
   - **Description**: Combined embedding and detection in one step.
   - **Parameters**:
     - `imgs`: Input images or video frames [B,C,H,W] or [F,C,H,W].
     - `msgs`: Messages to embed [B,K] (images) or [1,K] (video).
     - `is_video`: Whether input is a video.
   - **Returns**: Tuple of (watermarked_imgs, detected_predictions).

2. **embed(imgs, msgs, is_video=False)**
   - **Description**: Embeds messages into images or video frames.
   - **Parameters**: Same as forward().
   - **Returns**: Watermarked images/video [B,C,H,W] or [F,C,H,W].

3. **detect(imgs, is_video=False)**
   - **Description**: Detects messages from watermarked images or video.
   - **Parameters**: 
     - `imgs`: Watermarked images/video.
     - `is_video`: Whether input is a video.
   - **Returns**: Predictions tensor [B,(1+K),H,W] or [F,(1+K),H,W].

### Video-specific functions

4. **detect_video_and_aggregate(imgs, aggregation="avg")**
   - **Description**: Detects messages from video and aggregates across frames.
   - **Parameters**:
     - `imgs`: Watermarked video frames [F,C,H,W].
     - `aggregation`: Method to aggregate predictions ("avg", "squared_avg", "l1norm_avg", "l2norm_avg").
   - **Returns**: Binary message [1,K].

## Configuration parameters

You can adjust the following parameters to control the watermarking behavior:

### Model parameters (set at initialization)

- **scaling_w**: Controls the strength of the watermark (default: 0.2).
- **img_size**: Processing size for the model (default: 256).
- **clamp**: Whether to clamp output to [0,1] range (default: True).
- **do_attenuation**: Whether to apply the JND attenuation (default: True).
- **lowres_attenuation**: Whether to attenuate at low resolution (default: True).

### Video watermarking parameters

- **chunk_size**: Number of frames to process at once for videos (default: 16). Higher values may cause OOM errors.
- **step_size**: Interval between watermarked frames in video mode (default: 4).
- **video_mode**: Strategy for watermarking videos (default: "repeat").
  - Options: "repeat", "alternate", "interpolate".

### Runtime adjustments

You can modify some parameters after loading the model:

```python
# Adjust watermark strength.
model.blender.scaling_w = 0.1  # More subtle watermark
model.blender.scaling_w = 0.5  # Stronger watermark

# Attenuation at high resolution.
model.lowres_attenuation = False

# Adjust step size for video watermarking.
model.step_size = 2  # More robust, but slower embedding
model.step_size = 16  # Faster embedding, but a bit less robust
```

> [!TIP]
> You can set `model.clamp = False` and `model.do_attenuation = False`, then do `img_w - img` to get the watermarking residual that is predicted by the embedder.
```python
model.clamp = False
model.do_attenuation = False
```

## Understanding the outputs

When using the detection function, the model returns a tensor with the following structure:

- First channel (index 0): Detection mask (not used in current implementation).
- Remaining channels (indices 1 to K): Bit predictions.
  - Positive values indicate bit 1.
  - Negative values indicate bit 0.

For image watermarking, the predictions are directly interpreted. For video watermarking, predictions can be aggregated across frames using one of the available aggregation methods.
