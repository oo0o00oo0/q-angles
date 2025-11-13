import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# --------- CONFIG ---------
BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA_ANGLES = "dx8152/Qwen-Edit-2509-Multiple-angles"
LORA_LIGHTNING = "lightx2v/Qwen-Image-Lightning"  # recommended by the LoRA author

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("Using device:", DEVICE, "| dtype:", DTYPE)

# --------- LOAD PIPELINE ---------
pipe = QwenImageEditPlusPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE,
).to(DEVICE)

# memory helpers (good for 8–12 GB VRAM)
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# Load LoRAs
print("Loading Lightning LoRA...")
pipe.load_lora_weights(LORA_LIGHTNING, adapter_name="lightning")

print("Loading Angle LoRA...")
pipe.load_lora_weights(LORA_ANGLES, adapter_name="angles")

# Enable both adapters with weights (tweak these if you like)
pipe.set_adapters(
    ["lightning", "angles"],
    adapter_weights=[0.8, 1.0],
)

# --------- INPUT IMAGE ---------
input_path = "input.jpg"      # put your test image here
output_path = "output_angles.png"

print("Opening input image:", input_path)
init_img = Image.open(input_path).convert("RGB")

# --------- CAMERA COMMAND PROMPT ---------
# Example: move camera forward + wide angle
angle_prompt = "将镜头向前移动 将镜头转为广角镜头"

# Other phrases you can try / combine:
#   将镜头向左移动        (move left)
#   将镜头向右移动        (move right)
#   将镜头向下移动        (move down)
#   将镜头向左旋转45度    (rotate 45° left)
#   将镜头向右旋转45度    (rotate 45° right)
#   将镜头转为俯视        (top-down / bird view)
#   将镜头转为特写镜头    (close-up)
#
# Example alternatives:
# angle_prompt = "将镜头向左旋转45度"
# angle_prompt = "将镜头转为俯视 将镜头转为广角镜头"

generator = torch.Generator(device=DEVICE).manual_seed(0)

print("Running inference...")
result = pipe(
    image=[init_img],
    prompt=angle_prompt,
    negative_prompt="",
    num_inference_steps=4,       # very fast, like the Rapid AIO examples
    guidance_scale=1.0,          # Qwen likes low CFG
    true_cfg_scale=1.0,
    num_images_per_prompt=1,
    height=896,                  # safer than full 1024 on mid VRAM
    width=640,
    generator=generator,
).images[0]

result.save(output_path)
print("Saved:", output_path)
