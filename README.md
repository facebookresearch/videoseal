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
pip install -e . 
```

### Envs

- H2
    
    `/private/home/pfz/miniconda3/envs/img_wm`


### Launch simple experiment on Devfairs

To run on 2 GPUs:

`torchrun --nproc_per_node=2 train.py --local_rank 0`


### Launching experiments on SLURM cluster - clutils

To launch the expe of json file `expes/test/minimal.json`:

`python clutils/main.py sweep expes/test/minimal.json`

Should:
- Create the folder `/checkpoint/$USER/$YEAR_logs/minimal`
- Clone the repo in `/checkpoint/$USER/$YEAR_logs/minimal/code` corresponding to the branch `commit` of the json file (if the branch is not specified, it will clone the master branch).
- Run the experiments in the folder for every sweep of HPs.

More details in the clutils' [README](clutils/README.md).

