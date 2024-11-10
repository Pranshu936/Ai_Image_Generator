# model/diffusion_model.py
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
from PIL import Image
import base64

# Load the model
model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate image and convert it to base64
def generate_image(prompt):
    image = model(prompt).images[0]
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
