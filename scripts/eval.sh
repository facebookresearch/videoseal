#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=learnfair
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=6G
##SBATCH --constraint=volta32gb

EXP=${SLURM_ARRAY_TASK_ID}
OUTPUT_DIR_PREFIX="ffmpeg256px"
VIDEO_SIZE=256

if [[ ${EXP} = "0" ]]; then
    CKPT="/checkpoint/pfz/2024_logs/1122_vseal_04_rgb_96bits/_total_gnorm=1.0_sleepwake=False_scheduler=1_optimizer=AdamW,lr=1e-5_extractor_model=sam_small/checkpoint.pth"
    STEP_SIZE=4
    CHUNK_SIZE=32
    ATTENUATION=none
    OUTPUT_DIR="1122_vseal_04_rgb_96bits-default"
elif [[ ${EXP} = "1" ]]; then
    CKPT="/checkpoint/pfz/2024_logs/1122_vseal_04_rgb_96bits/_total_gnorm=1.0_sleepwake=False_scheduler=1_optimizer=AdamW,lr=1e-5_extractor_model=sam_small/checkpoint350.pth"
    STEP_SIZE=4
    CHUNK_SIZE=32
    ATTENUATION=none
    OUTPUT_DIR="1122_vseal_04_rgb_96bits-epoch350"
elif [[ ${EXP} = "2" ]]; then
    CKPT="/checkpoint/pfz/2024_logs/1122_vseal_04_rgb_96bits/_total_gnorm=1.0_sleepwake=False_scheduler=1_optimizer=AdamW,lr=1e-5_extractor_model=sam_small/checkpoint.pth"
    STEP_SIZE=4
    CHUNK_SIZE=32
    ATTENUATION=simplified_jnd_variance
    OUTPUT_DIR="1122_vseal_04_rgb_96bits-jnd_variance"
elif [[ ${EXP} = "3" ]]; then
    CKPT="/checkpoint/soucek/2025_logs/videoseal_jnd_variance_fixed2/_lambda_dec=1.0_lambda_d=0.0/checkpoint350.pth"
    STEP_SIZE=4
    CHUNK_SIZE=32
    ATTENUATION=simplified_jnd_variance
    OUTPUT_DIR="videoseal_jnd_variance_fixed2-epoch350"
elif [[ ${EXP} = "4" ]]; then
    CKPT="/checkpoint/soucek/2025_logs/videoseal_jnd_variance_fixed2/_lambda_dec=1.0_lambda_d=0.0/checkpoint700.pth"
    STEP_SIZE=4
    CHUNK_SIZE=32
    ATTENUATION=simplified_jnd_variance
    OUTPUT_DIR="videoseal_jnd_variance_fixed2-epoch700"
elif [[ ${EXP} = "5" ]]; then
    CKPT="/checkpoint/soucek/2025_logs/conv3d_embedder_temporal_extractor/expe/checkpoint.pth"
    STEP_SIZE=1
    CHUNK_SIZE=32
    ATTENUATION=none
    OUTPUT_DIR="conv3d_embedder_temporal_extractor-epoch95"
elif [[ ${EXP} = "6" ]]; then
    CKPT="/checkpoint/soucek/2025_logs/videoseal_continuation_onlyvideo/expe/checkpoint.pth"
    STEP_SIZE=4
    CHUNK_SIZE=32
    ATTENUATION=none
    OUTPUT_DIR="videoseal_continuation_onlyvideo-epoch145"
elif [[ ${EXP} = "7" ]]; then
    CKPT="/checkpoint/soucek/2025_logs/videoseal_continuation/expe/checkpoint200.pth"
    STEP_SIZE=4
    CHUNK_SIZE=32
    ATTENUATION=none
    OUTPUT_DIR="videoseal_continuation-epoch200"
fi



echo "Running eval of ${CKPT} with step_size: ${STEP_SIZE}, chunk_size: ${CHUNK_SIZE}, short_edge_size: ${VIDEO_SIZE}, and attenuation: ${ATTENUATION}."
echo "Results will be saved to ${OUTPUT_DIR}."

srun /private/home/soucek/.conda/envs/videoseal/bin/python -m videoseal.evals.full \
    --checkpoint ${CKPT} \
    --dataset sa-v --is_video true --num_samples 100 \
    --videowam_step_size ${STEP_SIZE} \
    --videowam_chunk_size ${CHUNK_SIZE} \
    --attenuation ${ATTENUATION} \
    --bdrate False \
    --save_first 10 \
    --short_edge_size ${VIDEO_SIZE} \
    --output_dir results/${OUTPUT_DIR_PREFIX}-${OUTPUT_DIR} \
    --simple_video_dataset True
    # --dataset coco --is_video false
