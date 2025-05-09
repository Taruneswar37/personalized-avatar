import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import get_peft_model, LoraConfig, TaskType

# Device
device = torch.device("cpu")

# === Dataset ===
class FaceDataset(Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)

dataset = FaceDataset("cropped_images")
dataloader = DataLoader(dataset, batch_size=1)

# === Load Stable Diffusion pipeline ===
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
unet = pipe.unet
pipe.to(device)

# === Wrap UNet with LoRA ===
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_k", "to_q"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)
unet = get_peft_model(unet, lora_config)

# === CLIP model for loss computation ===
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

optimizer = torch.optim.Adam(unet.parameters(), lr=1e-5)

# === Training ===
prompt = "a photo of the person"
for epoch in range(1):
    for i, face in enumerate(dataloader):
        face = face.to(device)

        # Generate image
        generated = pipe(prompt=prompt, image=face, strength=0.7, guidance_scale=7.5, num_inference_steps=25).images[0]
        generated = transforms.ToTensor()(generated).unsqueeze(0).to(device)

        # CLIP Loss
        inputs = clip_processor(text=[prompt], images=generated, return_tensors="pt", padding=True).to(device)
        outputs = clip(**inputs)
        similarity = outputs.logits_per_image
        loss = 1 - similarity.mean()  # Want high similarity → minimize (1 - score)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Step {i+1} Loss: {loss.item():.4f}")

# === Save LoRA Weights Only ===
unet.save_pretrained("lora_weights")
print("✅ LoRA training complete and weights saved.")
