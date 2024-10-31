# VideoSeal Demo

This directory hosts code for deploying the VideoSeal models to SageMaker using NVIDIA Triton.

## Setup

You'll need to create two conda environments from the root of this repo. The first environment will be used for development.

```
conda create -n videoseal-dev python=3.10 pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 conda-forge::ffmpeg -c pytorch -c nvidia
conda activate videoseal-dev
pip install -e .dev
```

The second environment will be used for the deployment. To keep the deployment light, it should only have the non-dev dependencies necessary for runtime. It cannot have editible dependencies (the `-e` flag in `pip install`), as they can't be packed via `conda-pack`.
```
conda create -n videoseal-deploy python=3.10 pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 conda-forge::ffmpeg -c pytorch -c nvidia
conda activate videoseal-deploy
pip install .
```

> [!NOTE]
> Whenever the inference code is updated, re-run `pip install .` in the `videoseal-deploy` environment to ensure that the latest changes are picked up for deployment.

## Deployment

The directory structure is set up with a subdirectory for each individual endpoint that we plan to host. Each endpoint subdirectory contains a [`model.py`](https://github.com/triton-inference-server/python_backend?tab=readme-ov-file#usage) implementation of a Triton Python backend as well as a [`config.pbtxt`](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) file defining the inputs, outputs and other configuration parameters for the endpoint.

```
endpoints
   <endpoint 1>
     model.py
     ...
   <endpoint 2>
   ...
```

### Build

Each endpoint subdirectory is also equipped with a `Makefile` that will build a Triton server directory. From the endpoint subdirectory, run:

```
BUILD_DIR=/scratch/$USER CONDA_ENV=video-deploy make
```

where the `$BUILD_DIR` env param declares where to build (this should be a non-NFS directory, like `/scratch/$USER` on FAIR Cluster) and the `$CONDA_ENV` env param declares which conda environment to pack along with the server.

### Test

To test the Triton server locally, `cd` into your build directory and run:

```
docker run \
  --tmpfs /tmp:exec \
  --mount type=bind,src=$(realpath ./models),dst=/mnt/models,readonly \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --gpus device=0 \
  --name $USER-tritonserver \
  calebyh/tritonserver:23.05.1-py3 \
  tritonserver --model-repository /mnt/models
```

> [!NOTE] You may have to add `sudo` to the above command if you don't have sufficient permissions.

Once the server is running, you can test against it with any prepared inputs in the endpoint's `test` directory.

```
python test/local.py
```

For example, an e2e test of the `embedder` endpoint might look like this:

```
(videoseal-dev) radkins@devfair0425:~/meta_apps/videoseal/videoseal/demo/endpoints/embedder$ python test/local.py
Sending request.json...
Response:
{'model_name': 'embedder',
 'model_version': '1',
 'outputs': [{'data': [<video_data>],
              'datatype': 'BYTES',
              'name': 'watermarked_video',
              'shape': [1]}]}
```

## Deploy

Follow the steps in [this doc](https://docs.google.com/document/d/1qVeYZSJL1unoPwWoUa_dQOORCM-P_kjuUy5-S0Ad6dM/edit#heading=h.6njyk6nmrqud) to deploy your local Triton build to SageMaker (using the `SSOOmnisealInferenceDev` role).