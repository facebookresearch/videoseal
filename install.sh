
conda create new installation python3.10
module load anaconda3/2023.03-1
module load cuda/12.2.0
module load cudnn/v9.0.0.312-cuda.12.2
conda install pytorch torchvision torchaudio pytorch-cuda=12.2 -c pytorch -c nvidia



conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda create --name videowm python==3.9

conda actviate videowm

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install hiera-transformer
pip install av
pip install PyWavelets
pip install pycocotools
pip install pandas 

git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 & pip install -e .

git clone https://github.com/fairinternal/omnisealbench.git
cd omnisealbench & pip install -e .
