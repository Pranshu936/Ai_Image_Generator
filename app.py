import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import PIL.Image
import io

# Load the model (you can adjust the model to one you want, e.g., Stable Diffusion)
@st.cache_resource
def load_model():
    return StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")

def pil_image_to_bytes(image: PIL.Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def apply_sepia_filter(image: PIL.Image.Image) -> PIL.Image.Image:
    # Convert to RGB if it's not already in RGB mode
    image = image.convert("RGB")
    sepia_image = PIL.Image.new("RGB", image.size)
    
    # Process each pixel to apply sepia filter
    pixels = image.load()
    sepia_pixels = sepia_image.load()
    for y in range(image.height):
        for x in range(image.width):
            r, g, b = pixels[x, y]

            # Apply sepia formula
            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)

            # Ensure values stay in [0, 255] range
            sepia_pixels[x, y] = (min(255, tr), min(255, tg), min(255, tb))

    return sepia_image
# Function to generate image from a prompt
def generate_image(prompt, resolution="512x512"):
    # Select resolution based on user input
    if resolution == "512x512":
        height, width = 512, 512
    elif resolution == "1024x1024":
        height, width = 1024, 1024
    elif resolution == "2048x2048":
        height, width = 2048, 2048

    # Generate image using the model
    pipe = load_model()
    image = pipe(prompt, height=height, width=width).images[0]
    return image

# Streamlit App
st.title("AI Image Generator 🚀")
st.write("Generate images from text descriptions with advanced options!")

# Sidebar for model settings
st.sidebar.title("Settings")
prompt = st.sidebar.text_area("Enter your text prompt:", "A beautiful sunset over the mountains")
resolution = st.sidebar.selectbox("Select Image Resolution:", ["512x512", "1024x1024", "2048x2048"])
num_images = st.sidebar.slider("Number of images to generate:", 1, 5, 1)
style_choice = st.sidebar.selectbox("Choose Image Style:", ["Realistic", "Abstract", "Cartoon", "Fantasy"])

# Image generation button
if st.sidebar.button("Generate Image"):
    st.write(f"Generating {num_images} image(s) in {style_choice} style...")
    
    # Loop through to generate multiple images
    for i in range(num_images):
        with st.spinner(f"Generating image {i+1}/{num_images}..."):
            generated_image = generate_image(f"{prompt}, {style_choice} style", resolution)
            st.image(generated_image, caption=f"Generated Image {i+1}", use_column_width=True)
            # Convert image to bytes for download
            image_bytes = pil_image_to_bytes(generated_image)
            st.download_button(
                "Download Image",
                data=image_bytes,
                file_name=f"generated_image_{i+1}.png",
                mime="image/png"
            )
    
    st.success(f"Image(s) generated successfully!")

# Optional Image-to-Image: Upload an image and apply transformation
st.sidebar.title("Image-to-Image Transformation")
uploaded_image = st.sidebar.file_uploader("Upload an image for transformation", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.sidebar.write("Image uploaded. Please wait while we transform it...")
    original_image = PIL.Image.open(uploaded_image)
    st.image(original_image, caption="Original Image", use_column_width=True)

    # Apply sepia transformation
    transformed_image = apply_sepia_filter(original_image)
    st.image(transformed_image, caption="Transformed Image (Sepia)", use_column_width=True)

    # Convert transformed image to bytes and add download button
    transformed_image_bytes = pil_image_to_bytes(transformed_image)
    st.download_button("Download Transformed Image", transformed_image_bytes, file_name="transformed_image.png", mime="image/png")

# Additional Interface Features
st.sidebar.markdown("""
    ## About this App
    This app generates images from textual descriptions using advanced AI models such as Stable Diffusion. 
    You can select the resolution, number of images, and styles to customize the output.

    Feel free to experiment with different text prompts, resolutions, and styles to create unique art.
""")
