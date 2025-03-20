# Videoseal

[[`Diary`](https://docs.google.com/document/d/1hQ8fd-ft1UAwsXCvlefA_MK3guwbVRyNmqhZMZy-0pQ/edit)]
<!-- [[`Method overview`]()] -->



## Setup

### Requirements

Version of Python is 3.10 (pytorch > 2.3, torchvision 0.16.0, torchaudio 2.1.0, cuda 12.1).
Install pytorch:
```
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Other dependencies:
```
pip install -r requirements.txt
```


### FFmpeg and VMAF

#### FFmpeg

Install FFmpeg based on your platform:

<details>
<summary>Mac</summary>

Install with [Homebrew](https://brew.sh/):
```
brew install ffmpeg
```
Then update the PATH:
```
export PATH=$PATH:/opt/homebrew/bin
```
Test the installation:
```
which ffmpeg
ffmpeg -version
```
It should output the path to the ffmpeg binary and the version of ffmpeg.

</details>

<details>
<summary>Linux</summary>

Install with apt:
```
sudo apt install ffmpeg
```
Test the installation:
```
which ffmpeg
ffmpeg -version
```
It should output the path to the ffmpeg binary and the version of ffmpeg.

</details>



#### VMAF
For VMAF score, install latest git build from [here](https://johnvansickle.com/ffmpeg/builds), then update the PATH:
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


### Envs

- H2
    
    `/private/home/pfz/miniconda3/envs/img_wm`

    Path to ffmpeg binary: `/private/home/pfz/09-videoseal/vmaf-dev/ffmpeg-git-20240629-amd64-static/ffmpeg`.
    To load the good binary, run  `export PATH=/private/home/pfz/09-videoseal/vmaf-dev/ffmpeg-git-20240629-amd64-static/ffmpeg:$PATH`



## Quick start for inference

### Download pre-trained model for 256-bits

For Linux:
```bash
wget https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth -P checkpoints/
```

For Mac:
```bash
mkdir checkpoints
curl -o checkpoints/y_256b_img.pth https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth
```

### Run inference

- On the cluster: Use the notebook [`notebooks/demos/video_inference.ipynb`](notebooks/demos/video_inference.ipynb)
- For Mac (local): Use [`notebooks/demos/video_inference_streaming.ipynb`](notebooks/demos/video_inference_streaming.ipynb) (optimized for lower RAM usage)

### Model details

- Default configuration: Uses model card `videoseal_1.0` with config for `y_256b_img.pth`
- Training configuration: See training grid at [img_train.json](https://github.com/fairinternal/videoseal/blob/main/expes/videoseal/img_train.json)


## Running experiments


### Launch simple experiment on Devfairs

To run on 2 GPUs:

`torchrun --nproc_per_node=2 train.py --local_rank 0`


### Launching experiments on SLURM cluster with clutils

To launch the expe of json file `expes/test/minimal.json`:

`python clutils/main.py sweep expes/test/minimal.json`

Should:
- Create the folder `/checkpoint/$USER/$YEAR_logs/minimal`
- Clone the repo in `/checkpoint/$USER/$YEAR_logs/minimal/code` corresponding to the branch `commit` of the json file (if the branch is not specified, it will clone the master branch).
- Run the experiments in the folder for every sweep of HPs.

More details in the clutils' [README](clutils/README.md).

### Logging

The logs are saved in the output folders in `log.txt` files.
The notebook `notebooks/plotlogs/plotlogs.ipynb` can be used to plot the logs.
