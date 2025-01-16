# VMAF

## What is VMAF?

VMAF is a perceptual video quality assessment algorithm developed by Netflix. It is based on the idea that the quality of a video is determined by the viewer's perception of the video, rather than the technical specifications of the video itself. VMAF uses a combination of machine learning and computer vision techniques to predict the perceived quality of a video based on a set of features extracted from the video.

For more information, see the [VMAF GitHub repository](https://github.com/Netflix/vmaf).

## Installation

You can check if VMAF is installed by running the following command:
```bash
ffmpeg -help
```
If `--enable-libvmaf` is in the output, VMAF is installed. Otherwise, you need to install it.

Install latest git build from [here](https://johnvansickle.com/ffmpeg/builds), then update the PATH:
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


## H2

Path to ffmpeg binary: `/private/home/pfz/09-videoseal/vmaf-dev/ffmpeg-git-20240629-amd64-static/ffmpeg`.
To load the good binary, run  `export PATH=$PATH:/private/home/pfz/09-videoseal/vmaf-dev/ffmpeg-git-20240629-amd64-static/ffmpeg`