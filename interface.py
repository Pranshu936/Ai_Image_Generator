# interface.py
import pynecone as pc
import requests

API_URL = "http://127.0.0.1:8000/generate-image"  # API endpoint

def submit_prompt(state):
    # Submit handler for generating the image
    prompt = state.prompt
    response = requests.post(API_URL, json={"prompt": prompt})
    if response.status_code == 200:
        state.generated_image = response.json()["image"]
    else:
        state.generated_image = None
        state.error_message = "Failed to generate image. Try again."

def app():
    # Web interface
    return pc.Form(
        pc.Input(placeholder="Enter your prompt", id="prompt", bind="prompt"),
        pc.Button("Generate", on_click="submit_prompt"),
        pc.Conditional(
            pc.Text("Image generation failed. Please try again.", id="error_message", visible="error_message"),
            pc.Image(src="data:image/png;base64,{generated_image}", visible="generated_image"),
        )
    )

# Start the app
if __name__ == "__main__":
    pc.run(app)
