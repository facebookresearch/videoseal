import os
import sys
import glob
import tqdm
import torch
import numpy as np
from einops import rearrange
from PIL import Image

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5
#python -m flux --name flux-schnell --height 512 --width 512 --prompt "A cat holding a sign that says hello world"


class FluxModel:

    def __init__(self, name="flux-schnell", device="cuda"):
        self.name = name
        self.device = device
        self.torch_device = torch.device(device)
        self.num_steps = 4 if name == "flux-schnell" else 50
        self.guidance = 3.5  # Guidance scale for denoising, can be adjusted

        self.t5 = load_t5(self.torch_device, max_length=256 if name == "flux-schnell" else 512)
        self.clip = load_clip(self.torch_device)
        self.model = load_flow_model(name, device=self.torch_device)
        self.ae = load_ae(name, device=self.torch_device)

    @torch.inference_mode()
    def denoise(self, image: Image, seed: int = 123, prompt: str = "A high definition image", step: int = -2):
        width, height = image.size
        height = 16 * (height // 16)
        width = 16 * (width // 16)
        if image.size != (width, height):
            print(f"Resizing image from {image.size} to {(width, height)}")
            image = image.resize((width, height))

        noise = get_noise(
            1,
            height,
            width,
            device=self.torch_device,
            dtype=torch.bfloat16,
            seed=123,
        )

        self.t5, self.clip = self.t5.to(self.torch_device), self.clip.to(self.torch_device)
        noise_input = prepare(self.t5, self.clip, noise, prompt=prompt)

        x = torch.from_numpy(np.array(image)).to(self.torch_device, dtype=torch.bfloat16).div_(127.5).sub_(1.0)
        x = rearrange(x, "h w c -> c h w").unsqueeze(0)
        with torch.autocast(device_type=self.torch_device.type, dtype=torch.bfloat16):
            x = self.ae.encode(x)
            x = x.to(torch.bfloat16)

        timesteps = get_schedule(self.num_steps, noise_input["img"].shape[1], shift=(self.name != "flux-schnell"))
        x_noised = x * (1 - timesteps[step]) + noise * timesteps[step]

        noised_image_input = prepare(self.t5, self.clip, x_noised, prompt=prompt)
        self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
        torch.cuda.empty_cache()

        img = noised_image_input["img"]
        img_ids = noised_image_input["img_ids"]
        txt = noised_image_input["txt"]
        txt_ids = noised_image_input["txt_ids"]
        vec = noised_image_input["vec"]
        
        guidance_vec = torch.full((1,), self.guidance, device=img.device, dtype=img.dtype) # this is ignored for schnell

        for t_curr, t_prev in zip(timesteps[step:][:-1], timesteps[step:][1:]):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            pred = self.model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            img = img + (t_prev - t_curr) * pred

        x = unpack(img.float(), height, width)
        with torch.autocast(device_type=self.torch_device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return img


if __name__ == "__main__":
    assert len(sys.argv) >= 2, "Usage: python thirdparty_watermark_removal.py <image_dir> [output_dir]"
    image_dir = sys.argv[1]
    output_dir = os.path.normpath(sys.argv[2] if len(sys.argv) > 2 else "output")

    strengths = [0, 3, 9, 15, 21] #list(range(25))[::3]
    for i in strengths:
        os.makedirs(output_dir + f"_{i:02d}", exist_ok=True)
    
    model = FluxModel(name="flux-dev", device="cuda")

    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    for image_file in tqdm.tqdm(image_files):
        image_ori = Image.open(image_file)
        
        # outputs = []
        for i in strengths:
            image = model.denoise(image=image_ori, step=-2 - i)
            image.save(os.path.join(output_dir + f"_{i:02d}", os.path.basename(image_file)))
            
            # outputs.append(image)
        # outputs = [image_ori.resize(outputs[0].size)] + outputs
        
        # output_image = Image.fromarray(np.concatenate(
        #     [np.array(img) for img in outputs], axis=1
        # )).save(os.path.join(output_dir, os.path.basename(image_file)))
