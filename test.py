import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np

# Load the inpainting model pipeline
model_id = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Move the pipeline to GPU for faster inference

# Load input image and mask
def load_image(image_path):
    """Load an image as a PIL object."""
    return Image.open(image_path).convert("RGB")

def load_mask(mask_path):
    """Load a binary mask image where the area to inpaint is white (255) and the rest is black (0)."""
    return Image.open(mask_path).convert("L")

image_path = "/root/3D-SDS/data/statue/images/IMG_2707.jpg"  # Replace with your image file
mask_path = "/root/3D-SDS/data/statue/seg/IMG_2707.jpg"    # Replace with your mask file

init_image = load_image(image_path)
mask_image = load_mask(mask_path)

# Define the inpainting prompt
prompt = "Inpaint to fit surrounding background"  # Customize the description

# Perform inpainting
result = pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=10).images[0]

# Save the result
output_path = "output_image.png"
result.save(output_path)
print(f"Inpainting completed. Result saved at {output_path}")
