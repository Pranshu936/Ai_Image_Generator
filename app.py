import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import PIL.Image
import io

# Load the Stable Diffusion model using caching to prevent reloading on every execution
@st.cache_resource
def load_model():
    # Load the pre-trained Stable Diffusion model from Hugging Face's model hub
    return StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")

# Convert a PIL image to bytes to enable downloading as a PNG file
def pil_image_to_bytes(image: PIL.Image.Image) -> bytes:
    # Create an in-memory bytes buffer
    buf = io.BytesIO()
    # Save the image to the buffer in PNG format
    image.save(buf, format="PNG")
    # Retrieve the byte data from the buffer
    return buf.getvalue()

# Apply a sepia filter to an image
def apply_sepia_filter(image: PIL.Image.Image) -> PIL.Image.Image:
    # Ensure image is in RGB mode for processing
    image = image.convert("RGB")
    # Create a new image with the same size for the sepia effect
    sepia_image = PIL.Image.new("RGB", image.size)
    
    # Load pixels of the original and new images
    pixels = image.load()
    sepia_pixels = sepia_image.load()
    
    # Loop through each pixel and apply sepia formula
    for y in range(image.height):
        for x in range(image.width):
            r, g, b = pixels[x, y]

            # Sepia tone calculation for red, green, and blue channels
            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)

            # Ensure values are within RGB range [0, 255]
            sepia_pixels[x, y] = (min(255, tr), min(255, tg), min(255, tb))

    return sepia_image

# Generate an image based on a text prompt and specified resolution
def generate_image(prompt, resolution="512x512"):
    # Set image resolution based on user selection
    if resolution == "512x512":
        height, width = 512, 512
    elif resolution == "1024x1024":
        height, width = 1024, 1024
    elif resolution == "2048x2048":
        height, width = 2048, 2048

    # Load the model pipeline and generate image from the prompt
    pipe = load_model()
    image = pipe(prompt, height=height, width=width).images[0]
    return image

# Streamlit App UI setup
st.title("AI Image Generator 🚀")
st.write("Generate images from text descriptions with advanced options!")

# Sidebar settings for image generation parameters
st.sidebar.title("Settings")
prompt = st.sidebar.text_area("Enter your text prompt:", "A beautiful sunset over the mountains")
resolution = st.sidebar.selectbox("Select Image Resolution:", ["512x512", "1024x1024", "2048x2048"])
num_images = st.sidebar.slider("Number of images to generate:", 1, 5, 1)
style_choice = st.sidebar.selectbox("Choose Image Style:", ["Realistic", "Abstract", "Cartoon", "Fantasy"])

# Image generation button
if st.sidebar.button("Generate Image"):
    st.write(f"Generating {num_images} image(s) in {style_choice} style...")
    
    # Generate multiple images if specified by the user
    for i in range(num_images):
        with st.spinner(f"Generating image {i+1}/{num_images}..."):
            # Generate image with prompt and selected style
            generated_image = generate_image(f"{prompt}, {style_choice} style", resolution)
            # Display generated image
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

# Optional Image-to-Image Transformation section
st.sidebar.title("Image-to-Image Transformation")
uploaded_image = st.sidebar.file_uploader("Upload an image for transformation", type=["jpg", "png", "jpeg"])

# If an image is uploaded, display and transform it
if uploaded_image is not None:
    st.sidebar.write("Image uploaded. Please wait while we transform it...")
    # Open uploaded image
    original_image = PIL.Image.open(uploaded_image)
    st.image(original_image, caption="Original Image", use_column_width=True)

    # Apply sepia transformation to the uploaded image
    transformed_image = apply_sepia_filter(original_image)
    st.image(transformed_image, caption="Transformed Image (Sepia)", use_column_width=True)

    # Convert transformed image to bytes for download
    transformed_image_bytes = pil_image_to_bytes(transformed_image)
    st.download_button("Download Transformed Image", transformed_image_bytes, file_name="transformed_image.png", mime="image/png")

# Additional information about the app in the sidebar
st.sidebar.markdown("""
    ## About this App
    This app generates images from textual descriptions using advanced AI models such as Stable Diffusion. 
    You can select the resolution, number of images, and styles to customize the output.

    Feel free to experiment with different text prompts, resolutions, and styles to create unique art.
""")
