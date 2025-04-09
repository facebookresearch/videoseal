# Watermark Removal

1. **Generate watermarked images for multiple methods.**

    You can use `videoseal/evals/full.py` script with option `--use_single_message True`. This will encode the same hidden message into all images and save the message into a `message.txt` file in the output directory. The same hidden message in all images is important as with different hidden messages, some attacks would not work.

    To replicate our experiments, just run the following script. It will run multiple watermarking baselines and save the watermarked images in `data/watermarked_images`.

    ```bash
    ./videoseal/quality_metric/generate_watermarked_images.sh
    ```

2. **Remove watermark using our _artifact discriminator_.**

    This can be done by running `videoseal/quality_metric/artifact_discriminator_watermark_removal.py` script.

    To replicate our experiments, just run the following script. It will run the artifact discriminator on `100` images from each folder in `data/watermarked_images`.

    ```bash
    DIRS=$(cd data/watermarked_images; echo *)
    for DIR in ${DIRS}; do
        echo ${DIR}
        python -m videoseal.quality_metric.artifact_discriminator_watermark_removal \
                  "data/watermarked_images/${DIR}" "data/watermarks_removed_ours/${DIR}" 100
    done
    ```

3. **Remove watermark using FLUX.**

    Remove watermark using [DiffPure](https://arxiv.org/abs/2205.07460) setup with [FLUX.1 [dev]](https://github.com/black-forest-labs/flux) model. For this, you need ~100GB VRAM, i.e., this requires, e.g., H200 GPU.

    Install [flux](https://github.com/black-forest-labs/flux) (you cannot use the HuggingFace implementation) and all its dependencies. Then, download `FLUX.1 [dev]` model weights from HuggingFace (this requires login and agreeing to T&C). You can just login via cli `huggingface-cli login` and the model weights will be downloaded automatically by the script if you agreed to T&C.

    The flux model can be run by `videoseal/quality_metric/thirdparty_watermark_removal.py` in the following sbatch script. The sbatch script can be run by `sbatch --array=1-5 script.sh`.

    ```bash
    #!/bin/bash
    #SBATCH --gpus=1
    #SBATCH --cpus-per-task=12
    #SBATCH --time=300
    #SBATCH --qos avseal_high

    DIRS=$(cd data/watermarked_images; echo *)
    DIR=$(echo ${DIRS} | cut -d' ' -f${SLURM_ARRAY_TASK_ID})
    srun python thirdparty_watermark_removal.py \
                "data/watermarked_images/${DIR}" "data/watermarks_removed_flux/${DIR}" 100
    ```

4. **Compute the metrics.**

    Once the watermarked images are in `data/watermarked_images/<wm_method_name>` and the images with removed watermarks in `data/watermarks_removed_<wm_removal_method_name>/<wm_method_name>`, the following script can be run to compute the metrics.

    ```bash
    python -m videoseal.quality_metric.eval_watermark_removal
    ```
