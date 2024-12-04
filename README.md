# :movie_camera: :seal: Video Seal: Open and Efficient Video Watermarking


This repository contains the official implementation of Videoseal, a state-of-the-art open-source video watermarking model that sets a new standard for efficiency and robustness. Our approach leverages temporal watermark propagation, a novel technique that converts any image watermarking model into an efficient video watermarking model, eliminating the need to watermark every frame in a video. We also propose a multistage training regimen that includes image pre-training, hybrid post-training, and extractor fine-tuning, supplemented with a range of differentiable augmentations. This repository includes pre-trained models, training code, inference code, baselines of state-of-the-art image watermarking models adapted for video watermarking (including MBRS, TrustMark, and WAM), and evaluation tools, all released under the MIT license, allowing for free use, modification, and distribution of the code and models.



| Original | Videoseal output | The watermark (nomalized for visibility)|
|---|---|---|
| <img src="./assets/_README_/1.gif" alt="example GIF" style="max-width: 100%; height: auto;"> | <img src="./assets/_README_/1_wmed.gif" alt="example GIF" style="max-width: 100%; height: auto;"> | <img src="./assets/_README_/1_wm.gif" alt="example GIF" style="max-width: 100%; height: auto;"> |
| <img src="./assets/_README_/2.gif" alt="example GIF" style="max-width: 100%; height: auto;"> | <img src="./assets/_README_/2_wmed.gif" alt="example GIF" style="max-width: 100%; height: auto;"> | <img src="./assets/_README_/2_wm.gif" alt="example GIF" style="max-width: 100%; height: auto;"> |
| <img src="./assets/_README_/3.gif" alt="example GIF" style="max-width: 100%; height: auto;"> | <img src="./assets/_README_/3_wmed.gif" alt="example GIF" style="max-width: 100%; height: auto;"> | <img src="./assets/_README_/2_wm.gif" alt="example GIF" style="max-width: 100%; height: auto;"> |
| <img src="./assets/_README_/4.gif" alt="example GIF" style="max-width: 100%; height: auto;"> | <img src="./assets/_README_/4_wmed.gif" alt="example GIF" style="max-width: 100%; height: auto;"> | <img src="./assets/_README_/4_wm.gif" alt="example GIF" style="max-width: 100%; height: auto;"> |
| <img src="./assets/_README_/5.gif" alt="example GIF" style="max-width: 100%; height: auto;"> | <img src="./assets/_README_/5_wmed.gif" alt="example GIF" style="max-width: 100%; height: auto;"> | <img src="./assets/_README_/5_wm.gif" alt="example GIF" style="max-width: 100%; height: auto;"> |


# Video Watermarking Using Video Seal  

<div style="background-color: #ff9900; padding: 20px; border-radius: 10px; font-family: 'Courier New', Courier, monospace; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">

```python
import torchvision.io
import videoseal
from videoseal.evals.metrics import bit_accuracy

# Load video and normalize to [0, 1]
video_path = "assets/videos/1.mp4"
video, _ = torchvision.io.read_video(video_path, output_format="TCHW")
video = video.float() / 255.0

# Load the model
model = videoseal.load("videoseal")

# Video Watermarking
outputs = model.embed(video, is_video=True) # this will embed a random msg
video_wmed = outputs["imgs_w"] # the watermarked video
msgs = outputs["msgs"] # the embedded message

# Extract the watermark message
msg_extracted = model.extract_message(imgs_w, aggregation="avg", is_video=True)

# VideoSeal can do Image Watermarking
img = video[0:1] # 1 x C x H x W
outputs = model.embed(img, is_video=False)
img_w = outputs["imgs_w"] # the watermarked image
msg_extracted = model.extract_message(imgs_w, aggregation="avg", is_video=False)
```
</div> 


## Installation

### Requirements

Version of Python is 3.10 (pytorch > 2.3, torchvision 0.16.0, torchaudio 2.1.0, cuda 12.1).
Install pytorch:

<div style="background-color: #f9900; padding: 20px; border-radius: 10px; font-family: 'Courier New', Courier, monospace; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">

```
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -e . 
```
</div>

#### VMAF

For VMAF score, install latest git build from [here](https://johnvansickle.com/ffmpeg/builds), then update the PATH:
<div style="background-color: #f9900; padding: 20px; border-radius: 10px; font-family: 'Courier New', Courier, monospace; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">

```
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz
tar -xvf ffmpeg-git-amd64-static.tar.xz 
export PATH=$PATH:/path/to/ffmpeg-git-20220307-amd64-static
```
Test the installation with:
```
which ffmpeg
ffmpeg -version
ffmpeg -filters | grep vmaf
```
It should output the path to the ffmpeg binary, the version of ffmpeg and the vmaf filter.
</div>



## Training and Running experiments


### Launch simple experiment a single machine with 2 GPUs 

<div style="background-color: #ff9900; padding: 20px; border-radius: 10px; font-family: 'Courier New', Courier, monospace; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">

```
torchrun --nproc_per_node=2 train.py --local_rank 0  
```
</div>

