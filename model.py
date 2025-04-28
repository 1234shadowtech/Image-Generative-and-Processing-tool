from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

# Load the Stable Diffusion 1.5 model
model_id = "runwayml/stable-diffusion-v1-5"

# Replace the scheduler with one compatible for CPU/GPU usage
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# Initialize the pipeline with the updated scheduler
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)

# Check for CUDA availability and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Move the pipeline to the appropriate device
pipe.to(device)

# Optimize memory usage based on device
pipe.enable_attention_slicing()  # Reduces memory usage
if device == "cuda":
    pipe.to(torch.float16)  # Use half-precision for GPU
else:
    pipe.to(torch.float32)  # Use full precision for CPU

# Reduce image resolution for faster processing (optional)
height = 384  # Smaller than default 512
width = 384

# Define the prompt
prompt = "red car"

# Generate the image
print("Generating image...")
image = pipe(prompt, height=height, width=width, num_inference_steps=25).images[0]

# Save the image
output_path = "output.png"
image.save(output_path)
print(f"Image saved to {output_path}")

