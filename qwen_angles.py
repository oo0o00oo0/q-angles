import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# Make the allocator less fragile
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA_ANGLES = "dx8152/Qwen-Edit-2509-Multiple-angles"
# We'll skip Lightning LoRA for now to reduce VRAM:
# LORA_LIGHTNING = "lightx2v/Qwen-Image-Lightning"

DTYPE = torch.float16

print("torch.cuda.is_available:", torch.cuda.is_available())

print("Loading base model with low_cpu_mem_usage=True...")
pipe = QwenImageEditPlusPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
)

# Let accelerate move blocks CPU <-> GPU automatically
pipe.enable_sequential_cpu_offload()
pipe.enable_vae_slicing()

print("Loading Angle LoRA...")
pipe.load_lora_weights(LORA_ANGLES, adapter_name="angles")
pipe.set_adapters(["angles"], adapter_weights=[1.0])

input_path = "input.jpg"
output_path = "output_angles.png"

print("Opening input image:", input_path)
init_img = Image.open(input_path).convert("RGB")

# Keep it small for the first working test
init_img = init_img.resize((512, 512))

angle_prompt = "将镜头向前移动 将镜头转为广角镜头"

generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)

print("Running inference (512x512, 4 steps)...")
result = pipe(
    image=[init_img],
    prompt=angle_prompt,
    negative_prompt="",
    num_inference_steps=4,
    guidance_scale=1.0,
    true_cfg_scale=1.0,
    num_images_per_prompt=1,
    height=512,
    width=512,
    generator=generator,
).images[0]

result.save(output_path)
print("Saved:", output_path)
EOF
