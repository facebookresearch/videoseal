#!/bin/bash
#SBATCH --job-name=omnisealbench_video
#SBATCH --gres=gpu:1
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


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# When using srun, the output and error files specified with --output and --error will only capture 
# the output and error from the srun command itself, not from the command being run by srun.
srun omnisealbench.evaluate \
    --eval_type video \
    --dataset "sa-v" \
    --dataset_dir "/large_experiments/meres/sa-v/sav_val_videos/" \
    --batch_size 1 \
    --num_workers 0 \
    --save_ids 0-9 \
    --postprocess_fn_device "cpu" \
    --skip_quality_metrics_on_attacks \
    --results_dir results/${OUTPUT_DIR_PREFIX}-${OUTPUT_DIR} \
    --model "videoseal" \
    --model__videoseal__additional_arguments__checkpoint ${CKPT} \
	--model__videoseal__additional_arguments__videowam_step_size ${STEP_SIZE} \
	--model__videoseal__additional_arguments__videowam_chunk_size ${CHUNK_SIZE} \
	--model__videoseal__additional_arguments__attenuation ${ATTENUATION} \
	--db__additional_arguments__output_resolution ${VIDEO_SIZE} \
    --watermark_override

## sbatch omnisealbench_eval.sh
