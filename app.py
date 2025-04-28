from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import color_space
import main
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

app = Flask(__name__)

# Configure directories
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/output/'
STYLE_FOLDER = 'image_styles/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize Stable Diffusion
model_id = "runwayml/stable-diffusion-v1-5"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)
pipe.enable_attention_slicing()
if device == "cuda":
    pipe.to(torch.float16)
else:
    pipe.to(torch.float32)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        output_image_path = None

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            operation = request.form.get('operation')
            style = request.form.get('style')

            if operation:
                try:
                    if operation == "resize":
                        width = int(request.form.get('width', 300))
                        height = int(request.form.get('height', 300))
                        output_image_path = color_space.resize(filepath, width, height)
                    elif operation == "rotate":
                        angle = int(request.form.get('angle', 90))
                        output_image_path = color_space.rotate(filepath, angle)
                    elif operation == "brightness":
                        factor = float(request.form.get('brightness_factor', 2.0))
                        output_image_path = color_space.brightness(filepath, factor)
                    elif operation == "contrast":
                        factor = float(request.form.get('contrast_factor', 1.5))
                        output_image_path = color_space.contrast(filepath, factor)
                    elif operation == "apply_blur":
                        radius = int(request.form.get('blur_radius', 2))
                        output_image_path = color_space.apply_blur(filepath, radius)
                    elif operation == "watermark":
                        text = request.form.get('watermark_text', 'Watermark')
                        output_image_path = color_space.add_watermark(filepath, text)
                    else:
                        # Handle other operations that don't require parameters
                        output_image_path = getattr(color_space, operation)(filepath)
                except Exception as e:
                    return f"Error processing image: {str(e)}"

            elif style:
                style_path = os.path.join(STYLE_FOLDER, style)
                output_image_path = os.path.join(OUTPUT_FOLDER, f"styled_{file.filename}")
                main.run_style_transfer(filepath, style_path, output_image_path)

        prompt = request.form.get('prompt')
        if prompt:
            output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], "generated_image.png")
            image = pipe(prompt, height=384, width=384, num_inference_steps=25).images[0]
            image.save(output_image_path)

        if output_image_path:
            return redirect(url_for('show_output', output_image=output_image_path.replace("static/", "")))

        return "Error: No operation selected or file not uploaded!"

    styles = os.listdir(STYLE_FOLDER)
    return render_template('index.html', styles=styles)

@app.route('/output')
def show_output():
    output_image = request.args.get('output_image', None)
    if output_image:
        return render_template('output.html', output_image=output_image)
    return "No output image to display!"

if __name__ == '__main__':
    app.run(debug=True)