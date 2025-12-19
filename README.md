<div align="center">

# 🎥 🦭 VideoSeal: SOTA Invisible Watermarking Models for Images & Videos

[![Demo](https://img.shields.io/badge/🌐_Demo-Try_Now-blue)](https://aidemos.meta.com/videoseal)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/videoseal/blob/main/notebooks/colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<table align="center">
  <tr>
    <td align="center"><b><a href="https://arxiv.org/abs/2412.09492">VideoSeal</a></b></td>
    <td align="center"><b><a href="#pixelseal-adversarial-only-training-for-invisible-watermarking">PixelSeal</a></b></td>
    <td align="center"><b><a href="https://arxiv.org/abs/2510.12812">ChunkySeal</a></b></td>
    <td align="center"><b><a href="https://arxiv.org/abs/2510.20468">WmForger</a></b></td>
  </tr>
  <tr>
    <td align="center"><a href="https://arxiv.org/abs/2412.09492"><img src="https://img.shields.io/badge/arXiv-2412.09492-b31b1b.svg" alt="arXiv"></a></td>
    <td align="center"><a href="https://ai.meta.com/research/publications/pixel-seal-adversarial-only-training-for-invisible-image-and-video-watermarking/"><img src="https://img.shields.io/badge/Paper-Meta_AI-0668E1.svg" alt="Paper"></a></td>
    <td align="center"><a href="https://arxiv.org/abs/2510.12812"><img src="https://img.shields.io/badge/arXiv-2510.12812-b31b1b.svg" alt="arXiv"></a></td>
    <td align="center"><a href="https://arxiv.org/abs/2510.20468"><img src="https://img.shields.io/badge/arXiv-2510.20468-b31b1b.svg" alt="arXiv"></a></td>
  </tr>
</table>

</div>

---

## 🔥 Highlights

<div align="center">
  <img src="docs/images/figure-pixelseal.png" width="100%" alt="PixelSeal Performance"/>
  <br>
  <em><b>PixelSeal</b> achieves state-of-the-art robustness-imperceptibility trade-off, positioning at the Pareto frontier</em>
</div>

<br>

- 🏆 **PixelSeal**: SOTA imperceptibility & robustness through adversarial-only training and JND-based attenuation
- 🚀 **ChunkySeal**: 4× capacity increase (1024 bits) - proving watermarking limits are far from reached
- 🎬 **VideoSeal**: Efficient image & video watermarking with temporal consistency
- 🔓 **Open Source**: All models, training code, and evaluation tools released under MIT license

---

## 📰 Latest Updates

- **December 2025**: 🆕 **ChunkySeal** and **PixelSeal** released! Model cards and checkpoints now available
- **October 2025**: 🏅 [**WmForger**](https://arxiv.org/abs/2510.20468) accepted to **NeurIPS 2025 as Spotlight**! Code in [`wmforger/`](wmforger/)
- **March 2025**: VideoSeal v1.0 with improved 256-bit model and enhanced robustness
- **December 2024**: Initial VideoSeal release with 96-bit baseline model


## 🚀 Quick start

```python
import videoseal
from PIL import Image
import torchvision.transforms as T

# Load any model by name (automatically downloads on first use)
model = videoseal.load("videoseal")     # VideoSeal v1.0 (256-bit, stable)
# model = videoseal.load("pixelseal")   # PixelSeal (SOTA imperceptibility & robustness)
# model = videoseal.load("chunkyseal")  # ChunkySeal (1024-bit high capacity)

# Watermark an image 🎨
img_tensor = T.ToTensor()(Image.open("image.jpg")).unsqueeze(0)
outputs = model.embed(img_tensor)
T.ToPILImage()(outputs["imgs_w"][0]).save("watermarked.jpg")

# Detect watermarks
detected = model.detect(img_tensor)
hidden_message = (detected["preds"][0, 1:] > 0).float()  # Binary message
```

**Video watermarking:**

```python
import videoseal
import torchvision

# Load and watermark video 🎬
model = videoseal.load("videoseal")
video, _, _ = torchvision.io.read_video("video.mp4")
video = video.permute(0, 3, 1, 2).float() / 255.0

outputs = model.embed(video, is_video=True)
watermarked = (outputs["imgs_w"] * 255).byte().permute(0, 2, 3, 1)
torchvision.io.write_video("watermarked.mp4", watermarked, fps=30)
```

> 💡 **For standalone usage without dependencies**, see our [TorchScript guide](docs/torchscript.md) for pre-compiled models.

## Installation


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

For training, we also recommend using decord:
```
pip install decord
```
Note that there may be some issues with installing decord: https://github.com/dmlc/decord/issues/213
Everything should be working without decord for inference, but there may be issues for training in this case.

## Model Zoo

We provide a comprehensive suite of watermarking models with different trade-offs between capacity, robustness, and imperceptibility.

| Model | Capacity | Best For | Model Card | Checkpoint | Paper | Status |
|:------|:--------:|:---------|:----------:|:----------:|:-----:|:------:|
| **PixelSeal** | 256 bits | **SOTA Robustness & Imperceptibility** | [`pixelseal.yaml`](videoseal/cards/pixelseal.yaml) | [pixelseal/checkpoint.pth](https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth) | [Link](#pixelseal-adversarial-only-training-for-invisible-watermarking) | 🆕 New |
| **ChunkySeal** | **1024 bits** | **High Capacity larger model** | [`chunkyseal.yaml`](videoseal/cards/chunkyseal.yaml) | [chunkyseal/checkpoint.pth](https://dl.fbaipublicfiles.com/videoseal/chunkyseal/checkpoint.pth) | [arXiv:2510.12812](https://arxiv.org/abs/2510.12812) | 🆕 New |
| **VideoSeal v1.0** | 256 bits | **Stable** | [`videoseal_1.0.yaml`](videoseal/cards/videoseal_1.0.yaml) | [y_256b_img.pth](https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth) | [arXiv:2412.09492](https://arxiv.org/abs/2412.09492) | ✅ Stable |
| VideoSeal v0.0 | 96 bits | Legacy Baseline | [`videoseal_0.0.yaml`](videoseal/cards/videoseal_0.0.yaml) | [rgb_96b.pth](https://dl.fbaipublicfiles.com/videoseal/rgb_96b.pth) | [arXiv:2412.09492](https://arxiv.org/abs/2412.09492) | 🟡 Legacy |

**Note**: For complete training checkpoints (with optimizer states and discriminators), see [docs/training.md](docs/training.md).


### Download the other models used as baselines

We do not own any third-party models, so you have to download them manually.
We provide a guide on how to download the models at [docs/baselines.md](docs/baselines.md).

### VMAF

We provide a guide on how to check and install VMAF at [docs/vmaf.md](docs/vmaf.md).






## Inference

### Notebooks

- [`notebooks/image_inference.ipynb`](notebooks/image_inference.ipynb)
- [`notebooks/video_inference.ipynb`](notebooks/video_inference.ipynb)
- [`notebooks/video_inference_streaming.ipynb`](notebooks/video_inference_streaming.ipynb): optimized for lower RAM usage

### Audio-visual watermarking

[`inference_av.py`](inference_av.py) 

To watermark both audio and video from a video file.
It loads the full video in memory, so it is not suitable for long videos.

Example:
```bash
python inference_av.py --input assets/videos/1.mp4 --output_dir outputs/
python inference_av.py --detect --input outputs/1.mp4
```

### Streaming embedding and extraction

[`inference_streaming.py`](inference_streaming.py) 

To watermark a video file in streaming.
It loads the video clips by clips, so it is suitable for long videos, even on laptops.

Example:
```bash
python inference_streaming.py --input assets/videos/1.mp4 --output_dir outputs/
```
Will output the watermarked video in `outputs/1.mp4` and the binary message in `outputs/1.txt`.


### Full evaluation

[`videoseal/evals/full.py`](videoseal/evals/full.py)

To run full evaluation of models and baselines.

Example to evaluate a trained model:
```bash
python -m videoseal.evals.full \
    --checkpoint /path/to/videoseal/checkpoint.pth \
```
or, to run a given baseline:
```bash
python -m videoseal.evals.full \
    --checkpoint baseline/wam \
``` 

This should save a file called `metrics.csv` with image/video imperceptibility metrics and the robustness to each augmentation (you can remove some of them to make the evaluation faster).
For instance, running the eval script for the default `videoseal` model on high-resolution videos from the SA-V dataset should give metrics similar to [sav_256b_metrics](https://dl.fbaipublicfiles.com/videoseal/sav_256b_metrics.csv).


## More details

### Training

We provide training code to reproduce our models or train your own models. This includes image and video training (we recommand training on image first, even if you wish to do video).
See [docs/training.md](docs/training.md) for detailed instructions on data preparation, training commands, and pre-trained model checkpoints.

### Inference

Here are some important parameters for the models:
* **`scaling_w`**: Controls the global watermark strength (default `0.2`). Higher values increase robustness against attacks but make the watermark more visible; lower values improve imperceptibility.
* **`attenuation`**: Enables Just Noticeable Difference (JND) masking. The JND model builds a heatmap that is high when there is a lot of texture, and low otherwise. It allows the model to hide stronger watermarks in these textured areas while preserving smooth regions. By default the `videoseal_1.0` model uses a JND heatmap (the one present in [modules/jnd.py](https://github.com/facebookresearch/videoseal/blob/main/videoseal/modules/jnd.py)).

You can also modify some model attributes after loading.
```python
# Example: updating parameters on an already loaded model
model.blender.scaling_w = 0.4   # Increase strength (more robust)
```

## License

The model is licensed under an [MIT license](LICENSE).

## Contributing

See [contributing](.github/CONTRIBUTING.md) and the [code of conduct](.github/CODE_OF_CONDUCT.md).

## See Also

- [**AudioSeal**](https://github.com/facebookresearch/audioseal)
- [**Watermark-Anything**](https://github.com/facebookresearch/watermark-anything/)

## Maintainers and contributors

Pierre Fernandez, Hady Elsahar, Tomas Soucek, Sylvestre Rebuffi, Alex Mourachko

## 📜 Papers & Citations

If you find this repository useful, please consider giving a star ⭐ and cite the relevant papers:

### VideoSeal: Open and Efficient Video Watermarking

[![arXiv](https://img.shields.io/badge/arXiv-2412.09492-b31b1b.svg)](https://arxiv.org/abs/2412.09492)

*Pierre Fernandez, Hady Elsahar, I. Zeki Yalniz, Alexandre Mourachko*

**Demo**: [aidemos.meta.com/videoseal](https://aidemos.meta.com/videoseal)

```bibtex
@article{fernandez2024videoseal,
  title={Video Seal: Open and Efficient Video Watermarking},
  author={Fernandez, Pierre and Elsahar, Hady and Yalniz, I. Zeki and Mourachko, Alexandre},
  journal={arXiv preprint arXiv:2412.09492},
  year={2024}
}
```

### ChunkySeal: We Can Hide More Bits

[![arXiv](https://img.shields.io/badge/arXiv-2510.12812-b31b1b.svg)](https://arxiv.org/abs/2510.12812)

*Aleksandar Petrov, Pierre Fernandez, Tomáš Souček, Hady Elsahar*

Despite rapid progress in deep learning-based image watermarking, the capacity of current robust methods remains limited to the scale of only a few hundred bits. This work establishes theoretical upper bounds on watermarking capacity and demonstrates **ChunkySeal**, which increases capacity 4× to **1024 bits** while preserving image quality and robustness.

```bibtex
@misc{petrov2025hidebits,
  title={We Can Hide More Bits: The Unused Watermarking Capacity in Theory and in Practice}, 
  author={Aleksandar Petrov and Pierre Fernandez and Tomáš Souček and Hady Elsahar},
  year={2025},
  eprint={2510.12812},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2510.12812}
}
```

### PixelSeal: Adversarial-Only Training for Invisible Watermarking

[**Paper**](https://ai.meta.com/research/publications/pixel-seal-adversarial-only-training-for-invisible-image-and-video-watermarking/)

*Tomáš Souček\*, Pierre Fernandez\*, Hady Elsahar, Sylvestre-Alvise Rebuffi, Valeriu Lacatusu, Tuan Tran, Tom Sander, Alexandre Mourachko*

This work introduces **adversarial-only training** that eliminates unreliable perceptual losses, achieving state-of-the-art robustness and imperceptibility. PixelSeal addresses optimization instability and resolution scaling challenges through a three-stage training schedule and JND-based attenuation.

```bibtex
@article{soucek2025pixelseal,
  title={Pixel Seal: Adversarial-only Training for Invisible Image and Video Watermarking},
  author={Souček, Tomáš and Fernandez, Pierre and Elsahar, Hady and Rebuffi, Sylvestre-Alvise and Lacatusu, Valeriu and Tran, Tuan and Sander, Tom and Mourachko, Alexandre},
  year={2025}
}
```

### WmForger: Black-Box Watermark Forging Attack

[![arXiv](https://img.shields.io/badge/arXiv-2510.20468-b31b1b.svg)](https://arxiv.org/abs/2510.20468)

*Tomáš Souček, Sylvestre-Alvise Rebuffi, Pierre Fernandez, Nikola Jovanović, Hady Elsahar, Valeriu Lacatusu, Tuan Tran, Alexandre Mourachko*

**NeurIPS 2025 Spotlight** 🏅 | [Virtual Site](https://neurips.cc/virtual/2025/loc/san-diego/poster/115131)

```bibtex
@article{soucek2025wmforger,
  title={Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models},
  author={Souček, Tomáš and Rebuffi, Sylvestre-Alvise and Fernandez, Pierre and Jovanović, Nikola and Elsahar, Hady and Lacatusu, Valeriu and Tran, Tuan and Mourachko, Alexandre},
  journal={arXiv preprint arXiv:2510.20468},
  year={2025}
}
```
