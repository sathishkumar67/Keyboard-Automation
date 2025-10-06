import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from capture import take_screenshot
import base64

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Load environment variables from the .env file
load_dotenv()

client = AzureOpenAI(
    api_version="2025-03-01-preview",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint="https://harsh-mamhtiwt-eastus2.cognitiveservices.azure.com/"  # Fixed: removed trailing spaces
)

# Take screenshot and get the image path
image_path = take_screenshot()

# Encode the image to base64
image_base64 = encode_image_to_base64(image_path)

response = client.chat.completions.create(
    model="gpt-4.1",  # Replace with your deployment name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ],
    max_tokens=150
)

print(response.choices[0].message.content)