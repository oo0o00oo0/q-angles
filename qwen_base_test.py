import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print("Using device:", DEVICE, "| dtype:", DTYPE)

# --- load base pipeline on GPU ---
pipe = QwenImageEditPlusPipeline.from_pretrained(
    BASE_MODEL,
    dtype=DTYPE,          # use bf16 on 4090
)
pipe.to(DEVICE)

# simple helpers
pipe.set_progress_bar_config(disable=None)

input_path = "input.jpg"
output_path = "output_base.png"

print("Opening input image:", input_path)
init_img = Image.open(input_path).convert("RGB")

# Qwen is happiest around 1024x1024
init_img = init_img.resize((1024, 1024))

prompt = "Move the camera slightly forward and make a subtle wide-angle view."

generator = torch.Generator(device=DEVICE).manual_seed(0)

print("Running base inference (1024x1024, 8 steps)...")
result = pipe(
    image=[init_img],
    prompt=prompt,
    negative_prompt="",
    num_inference_steps=8,
    guidance_scale=1.0,
    true_cfg_scale=1.0,
    num_images_per_prompt=1,
    height=1024,
    width=1024,
    generator=generator,
).images[0]

result.save(output_path)
print("Saved:", output_path)
