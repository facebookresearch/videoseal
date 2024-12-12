import argparse

import torch
import torchaudio
import torchvision
from audioseal import AudioSeal

import videoseal
from videoseal.utils.display import save_video_audio_to_mp4

parser = argparse.ArgumentParser(description="Video and Audio Watermarking")
parser.add_argument(
    "--input_file", type=str, required=True, help="Path to the input mp4 file"
)
parser.add_argument(
    "--output_file",
    type=str,
    required=False,
    default="watermarked.mp4",
    help="Path to save the watermarked mp4 file",
)
parser.add_argument(
    "--video_only", action="store_true", help="Watermark only the video, not the audio"
)
parser.add_argument(
    "--detect",
    action="store_true",
    help="Detect watermarks in the output video and audio",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the VideoSeal model
video_model = videoseal.load("videoseal")
video_model.eval()
video_model.to(device)
# Read the video and convert to tensor format
video, audio, info = torchvision.io.read_video(args.input_file, output_format="TCHW")

assert "audio_fps" in info, "The input video must contain an audio track. Simply refer to the main videoseal inference code if not."

fps = info["video_fps"]
sample_rate = info["audio_fps"]

# Normalize the video frames to the range [0, 1] and trim to 1 second
audio = audio.float()
video = video.float() / 255.0


if not args.detect:

    # Perform watermark embedding on video
    with torch.no_grad():
        outputs = video_model.embed(video, is_video=True)

    # Extract the results
    video_w = outputs["imgs_w"]  # Watermarked video frames
    video_msgs = outputs["msgs"]  # Watermark messages

    if not args.video_only:
        # Resample the audio to 16kHz for watermarking
        audio_16k = torchaudio.transforms.Resample(sample_rate, 16000)(audio)

        # If the audio has more than one channel, average all channels to 1 channel
        if audio_16k.shape[0] > 1:
            audio_16k_mono = torch.mean(audio_16k, dim=0, keepdim=True)
        else:
            audio_16k_mono = audio_16k

        # Add batch dimension to the audio tensor
        audio_16k_mono_batched = audio_16k_mono.unsqueeze(0)

        # Load the AudioSeal model
        audio_model = AudioSeal.load_generator("audioseal_wm_16bits")

        # Get the watermark for the audio
        with torch.no_grad():
            audio_msg = torch.randint(
                0,
                2,
                (audio_16k_mono_batched.shape[0], audio_model.msg_processor.nbits),
                device=audio_16k_mono_batched.device,
            )
            watermark = audio_model.get_watermark(
                audio_16k_mono_batched, 16000, message=audio_msg
            )

        # Embed the watermark in the audio
        audio_16k_w = audio_16k_mono_batched + watermark

        # Remove batch dimension from the watermarked audio tensor
        audio_16k_w = audio_16k_w.squeeze(0)

        # If the original audio had more than one channel, duplicate the watermarked audio to all channels
        if audio_16k.shape[0] > 1:
            audio_16k_w = audio_16k_w.repeat(audio_16k.shape[0], 1)

        # Resample the watermarked audio back to the original sample rate
        audio_w = torchaudio.transforms.Resample(16000, sample_rate)(audio_16k_w)
    else:
        audio_w = audio

    # Save the watermarked video and audio

    save_video_audio_to_mp4(
        video_tensor=video,
        audio_tensor=audio_w,
        fps=int(fps),
        audio_sample_rate=int(sample_rate),
        output_filename=args.output_file,
    )

    print(f"encoded message: \n Audio: {audio_msg} \n Video {video_msgs[0]}")

else:
    # Detect watermarks in the video
    with torch.no_grad():
        msg_extracted = video_model.extract_message(video)
    print(f"Extracted message from video: {msg_extracted}")

    if not args.video_only:
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(0)  # batchify

        # if stereo convert to mono
        if audio.shape[1] > 1:
            audio = torch.mean(audio, dim=1, keepdim=True)

        # Load the AudioSeal detector model
        detector = AudioSeal.load_detector("audioseal_detector_16bits")

        # Detect watermarks in the audio
        with torch.no_grad():
            result, message = detector.detect_watermark(
                torchaudio.transforms.Resample(sample_rate, 16000)(audio), 16000
            )
        print(f"Detection result for audio: {result}")
        print(f"Extracted message from audio: {message}")
