import os
import base64
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def encode_image_to_data_uri(image_path: str, mime_type: str = "png") -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime_type};base64,{b64}"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
def get_code(query: str, screen_shot) -> str:
    screen_shot = encode_image_to_data_uri(screen_shot)
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
        {
            "role": "system",
            "content": "You are an agent that receives the current state of the environment and a task description as input. \\n\\n1. Input format example: Overall state: you are on a new empty tab. Task: open youtube \\n\\n2. Your job is to break the task into step-by-step actions, each written on a new line. \\n\\n3. The order of the actions must reflect how a human would typically perform the task. \\n\\n4. Only return the actions, nothing else (no explanations or extra text). \\n\\n5. For those actions, return a runnable Python script and nothing else. Output only Python source code (no surrounding text, no comments, no explanations, no backticks, no metadata). The code must be valid Python and executable as-is (include necessary imports, functions, and a runnable entry point if appropriate). Avoid unnecessary verbosity in the code; prefer clear, minimal, runnable implementations. If the request cannot be fulfilled safely or would violate policy, briefly return a single-line Python comment explaining the refusal (still no extra prose). You only use the PyAutoGUI library; always import it at the start of the program and for now you will give only keyboard actions"
        },
        {
            "role": "user",
            "content": [
                        {
                            "type": "text",
                            "text": query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": screen_shot
                            }
                        }]
        }],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None
    )
    full_response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
    return full_response