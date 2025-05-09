import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

device = torch.device("cpu")

# Load fine-tuned pipeline
pipeline = StableDiffusionPipeline.from_pretrained("./lora_weights", torch_dtype=torch.float32)
pipeline.to(device)

# Input prompt
prompt = "a regal king wearing ornate royal robes, golden crown, majestic background, face completely visible, 4k, relaistic image"

# Generate image
image = pipeline(prompt=prompt).images[0]
image.save("generated_avatar.png")
print("Avatar saved to generated_avatar.png")
