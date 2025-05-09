**Personalized Avatar Generator**

This project enables the generation of personalized avatars by applying advanced image generation techniques. The pipeline combines several AI models, including MTCNN for face detection, InsightFace for face alignment, ControlNet for pose guidance, and Stable Diffusion with LoRA for image generation. Users can upload an image and provide a prompt to create an avatar that retains their face identity while transforming according to the prompt.

**Features**

Face Detection: Automatically detects and crops faces from images using MTCNN.

Face Alignment: Uses InsightFace to align faces for better identity preservation.

Pose Guidance: Leverages ControlNet and OpenPose to apply pose guidance based on face alignment.

Avatar Generation: Generates a personalized avatar using Stable Diffusion, guided by the provided prompt.

LoRA Fine-Tuning: Applies LoRA (Low-Rank Adaptation) for efficient fine-tuning of Stable Diffusion to produce high-quality avatars.

**Requirements**

Python 3.8+
PyTorch
Hugging Face's transformers
diffusers
Gradio for web-based UI
facenet-pytorch
insightface
controlnet_aux
peft
Other dependencies listed in requirements.txt

**Setup**

1. Clone the Repository
2. Install Dependencies
3. Download Pre-trained Models:
   The project requires several pre-trained models:
        Stable Diffusion: You can use the "runwayml/stable-diffusion-v1-5" model.
        InsightFace: You can download the model using insightface for face detection and alignment.
        ControlNet: The OpenPose model (lllyasviel/sd-controlnet-openpose) is used for pose detection.
4. LoRA Weights:
   If you have pre-trained LoRA weights, place them in the directory where the script expects them (e.g., lora_weights/). Otherwise, you can train LoRA weights by running the training script.


**Folder Structure**

├── avatar_generator.py        # Main script for avatar generation
├── train_lora.py              # Script to train LoRA weights
├── lora_weights/              # Folder to store trained LoRA weights
├── images/                    # Folder for input images
├── cropped_images/            # Folder to store cropped faces
├── requirements.txt           # Python dependencies
└── README.md                  # This file



