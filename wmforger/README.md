<div align="center">
<h1>🖼️🔓 Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models</h1>

**[NeurIPS 2025 (Spotlight 🏅)](https://arxiv.org/abs/2510.20468)**

<img src="https://dl.fbaipublicfiles.com/wmforger/overview.jpg" style="width:80%"/>

**[[arXiv](https://arxiv.org/abs/2510.20468)]**
**[[Colab](https://colab.research.google.com/github/facebookresearch/videoseal/blob/main/wmforger/example.ipynb)]**

</div>

## 🎮 Usage

See and try our example [jupyter notebook](https://github.com/facebookresearch/videoseal/blob/main/wmforger/example.ipynb) in [Colab](https://colab.research.google.com/github/facebookresearch/videoseal/blob/main/wmforger/example.ipynb) or try the mimimal example of removing watermark localy using the steps below.

1. Install [PyTorch](https://pytorch.org/get-started/) and other [requirements](https://github.com/facebookresearch/videoseal/blob/main/requirements.txt).

2. Clone the repository.
   ```sh
   git clone https://github.com/facebookresearch/videoseal.git
   cd videoseal/wmforger/
   ```

3. Download the pretrained model weights.
   ```sh
   wget https://dl.fbaipublicfiles.com/wmforger/convnext_pref_model.pth
   ```

4. Extract watermark.
   ```sh
   python optimize_image.py --ckpt_path convnext_pref_model.pth --image assets/tahiti_watermarked.png
   ```


## 🚆 Train preference model from scratch
1. Download SA-1b dataset.

2. Update path to the dataset in `configs/datasets/sa-1b-full.yaml`

3. Train. We trained using 8 GPUs.
   ```sh
   sbatch train-slurm.sh
   ```


## 🧾 License
Please see the LICENSE file in the root of the main repository.


## ✍️ Citation
If you find this repository useful, please consider giving a star ⭐ and please cite as:


```bibtex
@inproceedings{soucek2025transferable,
  title={Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models},
  author={Sou\v{c}ek, Tom\'{a}\v{s} and Rebuffi, Sylvestre-Alvise and Fernandez, Pierre and Jovanovi\'{c}, Nikola and Elsahar, Hady and Lacatusu, Valeriu and Tran, Tuan and Mourachko, Alexandre},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
