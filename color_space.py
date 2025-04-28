from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np

def gray_color(image_path):
    """Convert image to grayscale."""
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_path = image_path.replace("uploads", "output").replace(".jpg", "_gray.jpg")
    cv2.imwrite(output_path, gray_image)
    return output_path

def resize(image_path, width=300, height=300):
    """Resize image to specified dimensions."""
    try:
        image = Image.open(image_path)
        resized_image = image.resize((width, height))
        output_path = image_path.replace("uploads", "output").replace(".jpg", "_resized.jpg")
        resized_image.save(output_path)
        return output_path
    except Exception as e:
        raise Exception(f"Error resizing image: {str(e)}")

def rotate(image_path, angle=90):
    """Rotate image by specified angle."""
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    output_path = image_path.replace("uploads", "output").replace(".jpg", "_rotated.jpg")
    cv2.imwrite(output_path, rotated_image)
    return output_path

def brightness(image_path, factor=2.0):
    """Adjust image brightness."""
    image = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(image)
    brightened_image = enhancer.enhance(factor)
    output_path = image_path.replace("uploads", "output").replace(".jpg", "_brightened.jpg")
    brightened_image.save(output_path)
    return output_path

def contrast(image_path, factor=1.5):
    """Adjust image contrast."""
    image = Image.open(image_path)
    enhancer = ImageEnhance.Contrast(image)
    contrasted_image = enhancer.enhance(factor)
    output_path = image_path.replace("uploads", "output").replace(".jpg", "_contrast.jpg")
    contrasted_image.save(output_path)
    return output_path

def apply_blur(image_path, radius=2):
    """Apply Gaussian blur using PIL."""
    image = Image.open(image_path)
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius))
    output_path = image_path.replace("uploads", "output").replace(".jpg", "_blur.jpg")
    blurred_image.save(output_path)
    return output_path

def gaussian_blur(image_path, kernel_size=5):
    """Apply Gaussian blur using OpenCV."""
    image = cv2.imread(image_path)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    output_path = image_path.replace("uploads", "output").replace(".jpg", "_gaussian_blur.jpg")
    cv2.imwrite(output_path, blurred_image)
    return output_path

def sepia(image_path):
    """Apply sepia filter."""
    image = Image.open(image_path)
    width, height = image.size
    pixels = image.load()
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)
            pixels[x, y] = (min(tr, 255), min(tg, 255), min(tb, 255))
    output_path = image_path.replace("uploads", "output").replace(".jpg", "_sepia.jpg")
    image.save(output_path)
    return output_path

def sharpen(image_path, factor=2.0):
    """Sharpen the image."""
    image = Image.open(image_path)
    sharpened = image.filter(ImageFilter.SHARPEN)
    output_path = image_path.replace("uploads", "output").replace(".jpg", "_sharp.jpg")
    sharpened.save(output_path)
    return output_path

def edge_detect(image_path):
    """Apply edge detection."""
    image = cv2.imread(image_path)
    edges = cv2.Canny(image, 100, 200)
    output_path = image_path.replace("uploads", "output").replace(".jpg", "_edges.jpg")
    cv2.imwrite(output_path, edges)
    return output_path

def invert(image_path):
    """Invert image colors."""
    image = Image.open(image_path)
    inverted_image = ImageOps.invert(image)
    output_path = image_path.replace("uploads", "output").replace(".jpg", "_inverted.jpg")
    inverted_image.save(output_path)
    return output_path

def add_watermark(image_path, text="Watermark"):
    """Add text watermark to image."""
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    output_path = image_path.replace("uploads", "output").replace(".jpg", "_watermark.jpg")
    cv2.imwrite(output_path, image)
    return output_path

def mirror(image_path):
    """Mirror the image horizontally."""
    image = Image.open(image_path)
    mirrored = ImageOps.mirror(image)
    output_path = image_path.replace("uploads", "output").replace(".jpg", "_mirror.jpg")
    mirrored.save(output_path)
    return output_path