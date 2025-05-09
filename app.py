import gradio as gr
import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
from insightface.app import FaceAnalysis
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector

# Set device
device = torch.device("cpu")

# Load InsightFace for face alignment
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load ControlNet + OpenPose
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float32)
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# Load SD pipeline + LoRA weights
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Load LoRA weights if available
try:
    pipe.load_lora_weights("lora_weights", weight_name="pytorch_lora_weights.safetensors")
    pipe.fuse_lora()
except Exception as e:
    print(f"⚠️ Failed to load LoRA: {e}")

pipe.to(device)

# Inference function
def generate_avatar(input_image: Image.Image, prompt: str):
    input_np = np.array(input_image.convert("RGB"))
    faces = face_app.get(input_np)
    if not faces:
        return "❌ No face detected.", None

    aligned = faces[0].aligned
    if aligned is None:
        box = faces[0].bbox.astype(int)
        aligned = input_np[box[1]:box[3], box[0]:box[2]]
    
    aligned_pil = Image.fromarray(aligned)

    # Run pose detection
    try:
        pose = pose_detector(aligned_pil)
        pose = pose.resize(aligned_pil.size)
    except Exception as e:
        return f"❌ Pose detection failed: {e}", None

    # Generate the avatar
    try:
        result = pipe(
            prompt=prompt,
            image=aligned_pil,
            control_image=pose,
            strength=0.5,
            guidance_scale=8.5,
            num_inference_steps=50,
        ).images[0]
        return "✅ Success!", result
    except Exception as e:
        return f"❌ Generation failed: {e}", None


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Personalized Avatar Generator")
    gr.Markdown("Upload your photo and type a creative prompt!")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            prompt = gr.Textbox(lines=2, placeholder="e.g. person as a superhero with city background")
            submit = gr.Button("Generate Avatar")
        with gr.Column():
            status = gr.Textbox(label="Status")
            output_image = gr.Image(label="Generated Avatar")

    submit.click(fn=generate_avatar, inputs=[input_image, prompt], outputs=[status, output_image])

demo.launch(server_name="127.0.0.1", server_port=8888)
