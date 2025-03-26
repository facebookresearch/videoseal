# Clutils Examples

This directory contains example scripts that demonstrate how to use models with both local execution and distributed training on the cluster using Clutils.

## Available Examples

- **extract_fts.py**: Extract features from images using various models
- **finetune.py**: Finetune vision backbones on ImageNet with DDP support

## Running Locally on DevFair

### Feature Extraction

To extract features from images using a pre-trained model locally:

```bash
python extract_fts.py \
  --output_dir /path/to/output \
  --data_dir /path/to/images \
  --model_name dinov2_vits14 \
  --batch_size 64 \
  --resize_size 288
```

### Finetuning

To finetune a model locally using DDP (Distributed Data Parallel):

```bash
# For single GPU training
python finetune.py \
  --output_dir /path/to/output \
  --data_dir /path/to/imagenet \
  --model_name resnet50 \
  --batch_size 64 \
  --epochs 30 \
  --lr 0.01

# For multi-GPU training with torchrun (2 GPUs example)
OMP_NUM_THREADS=40 torchrun --nproc_per_node=2 finetune.py \
  --output_dir /path/to/output \
  --data_dir /path/to/imagenet \
  --model_name resnet50 \
  --batch_size 128 \
  --epochs 30 \
  --lr 0.01
```

## Running on Cluster with Clutils

### Configuration Files

Configuration files are used to specify the parameters for the experiments. The configuration files are written in JSON format and contain several important sections:

1. **Basic configuration**: Command to run, git repo, and environment setup
2. **Parameters**: The hyperparameters to sweep over
3. **Machine configuration**: SLURM settings for the cluster
4. **Meta information**: Job names and output directories

Here's an example configuration file for feature extraction (`extract.json`):

```json
{
    "cmd": "python -m examples.extract",
    "git": "git@github.com:fairinternal/clutils.git",
    "preload": "MKL_THREADING_LAYER=GNU",
    "branch": "main",
    "params": {
        "data_dir": "/datasets01/imagenet_full_size/061417",
        "model_name": ["dinov2_vits14", "dinov2_vitb14"],
        "resize_size": 280,
        "batch_size": 64,
        "chunk": 10,
        "chunk_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    },
    "machine": {
        "constraint": "volta32gb",
        "gres": "gpu:1",
        "nodes": "1",
        "tasks-per-node": "1",
        "partition": "learnlab",
        "cpus-per-task": "10",
        "time": 4320,
        "mem-per-cpu": "6G"
    },
    "meta": {
        "group": "", 
        "name": "2503_extract", 
        "dest-arg": "yes",
        "dest-name": "output_dir"
    }
}
```

And here's an example for finetuning (`finetune.json`):

```json
{
    "cmd": "python -m examples.finetune",
    "git": "git@github.com:fairinternal/clutils.git",
    "preload": "MKL_THREADING_LAYER=GNU",
    "branch": "main",
    "params": {
        "data_dir":     "/datasets01/imagenet_full_size/061417",
        "val_dir":      "/datasets01/imagenet_full_size/061417/val",
        "lr":           [0.01, 0.001],
        "model_name": {
            "resnet50":             {"wd": 1e-4},
            "vit_base_patch16_224": {"wd": 1e-2},
            "dinov2_vits14":        {"wd": 1e-2},
            "dinov2_vitb14":        {"wd": 1e-2}
        },
        "batch_size": 1024,
        ...
    },
    "machine": {
        "constraint": "volta32gb",
        "gres": "gpu:8",
        "nodes": "1",
        "partition": "learnlab",
        "cpus-per-task": "10",
        "time": 2880,
        "mem-per-cpu": "8G"
    },
    "meta": {
        "group": "", 
        "name": "2503_imagenet_finetune", 
        "dest-arg": "yes",
        "dest-name": "output_dir"
    }
}
```
Using dictionaries for `model_name` allows you to specify additional hyperparameters for each model.

#### Key configuration parameters

- **git**: The GitHub repository to clone (e.g., `git@github.com:fairinternal/clutils.git`). Clutils will clone this repo and run your script from there.
  
- **branch**: The git branch to use (default is `main`). You can specify a different branch to test code that's not in the main branch.

- **dest-name**: This parameter defines the argument in your script corresponding to the output directory, where results will be saved. Clutils will override this argument for each job in your sweep and pass its path to your script.

- **name**: The experiment name for this sweep (e.g., `2503_imagenet_finetune`). It's recommended to prefix this with the date (MMDD format) for better organization and tracking of experiments.

### Parameter Sweeps

The `params` section defines the hyperparameters for your sweep. When you provide a list for a parameter, Clutils will create a separate job for each value. For example, in the finetuning configuration above:

```json
"params": {
    "model_name": [
        "resnet50",
        "vit_base_patch16_224",
        "dinov2_vits14",
        "dinov2_vitb14"
    ],
    "lr": [0.01, 0.001]
}
```

This will create 8 jobs (4 models × 2 learning rates).

### Launching Experiments

To launch experiments on the cluster:

```bash
# Count the number of jobs in the sweep
clutils count finetune.json

# Launch the experiment sweep
clutils sweep finetune.json

# Check the status of your jobs
clutils status

# Check progress and results of experiments
clutils check
```

### Output directory structure

After launching a sweep, Clutils will create a directory structure like:

```
/checkpoint/<username>/2023_logs/2503_imagenet_finetune/
├── code/                  # Cloned repository
├── logs/                  # Output logs
│   ├── _model_name=resnet50_lr=0.01.stdout
│   ├── _model_name=resnet50_lr=0.01.stderr
│   └── ...
├── _model_name=resnet50_lr=0.01/  # Output directory for this configuration
│   ├── args.json
│   ├── checkpoint_best.pth
│   └── ...
├── ...                    # More job directories
├── run.sh                 # Generated SLURM submission script
├── params.txt             # Parameter combinations
├── commands.txt           # Generated commands
└── status.txt             # Status tracking file
```

Each job will have its own output directory where results will be saved.
