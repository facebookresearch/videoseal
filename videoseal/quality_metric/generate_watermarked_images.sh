MODELS="VideoSealv1:videoseal_0.0
TrustMark:baseline/trustmark
WAM:baseline/wam
CIN:baseline/cin
MBRS:baseline/mbrs
HIDDEN:baseline/hidden
VideoSealv2pp:videoseal_1.0_128bits
VideoSealv2pp256bit:videoseal_1.0
VideoSealExp1:/checkpoint/pfz/2025_logs/0219_vseal_convnextextractor/_nbits=128_lambda_i=0.1_embedder_model=1/checkpoint600.pth --scaling_w 0.016
VideoSealv2:/checkpoint/soucek/2025_logs/0228_vseal_128bits_jnd_ftvid_complete/_optimizer=AdamW,lr=1e-5_videowam_step_size=1/checkpoint200.pth --lowres_attenuation True
VideoSealv2image:/checkpoint/soucek/2025_logs/0303_vseal_ydisc_mult1_bis_ft-fixed-lr5e6/expe/checkpoint200.pth --lowres_attenuation True
VideoSealCVVDP:/checkpoint/soucek/2025_logs/videoseal_cvvdp_sweep/_lambda_i=0.5/checkpoint500.pth"

mapfile -t lines <<< "$MODELS"

for LINE in "${lines[@]}"; do
    KEY=$(echo $LINE | cut -f1 -d:)
    CKPT=$(echo $LINE | cut -f2 -d:)
    mkdir -p data/watermarked_images/${KEY}
    python -m videoseal.evals.full --dataset sa-1b-full \
                                   --is_video false \
                                   --num_samples 1000 \
                                   --save_first 1000 \
                                   --output_dir data/watermarked_images/${KEY} \
                                   --decoding False \
                                   --skip_image_metrics True \
                                   --use_single_message True \
                                   --checkpoint ${CKPT}
done
