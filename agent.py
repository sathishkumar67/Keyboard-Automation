from __future__ import annotations
import os
from groq import Groq
from dotenv import load_dotenv
from capture import take_screenshot
from utils import encode_image_to_data_uri
from programmer import get_code, run_code


CONTEXT = [{
            "role": "system",
            "content": "You are an AI agent that decides tasks and invokes tools when needed.  \nYou have access to the following tool:  \n\nTool: action  \nDescription: Executes keyboard actions such as typing text, pressing keys, or key combinations.  \n\nRules:\n- If a task requires a keyboard action, output a JSON object in the format:\n  {\"tool\": \"action\", \"input\": \"<keyboard action description including where to type or press>\"}\n- The \"input\" field must always include both the action and the location, e.g., \"type 'hello world' in the search box\" or \"press Enter key\".\n- Otherwise, just output a plain text response.\n- Do not explain the tool invocation, only return the JSON when calling the tool.\n"
        }]

# Load environment variables from the .env file
load_dotenv()



# Initialize the Groq client with the API key from environment variables
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def perform_task(query: str):
    # take current screenshot
    screen_shot = encode_image_to_data_uri(take_screenshot())
    messages = CONTEXT + [{
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
                }
            ]
        }]

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=messages,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None
    )

    response = completion.choices[0].message.content
    print("Model response:", response)

    # --- Now detect tool call ---
    import json
    try:
        parsed = json.loads(response)
        if parsed.get("tool") == "action":
            # Call your tool
            run_code(parsed['input'], screen_shot)
            print(f"Invoking tool 'action' with input: {parsed['input']}")
        else:
            print("AI response:", response)
    except:
        print("AI response:", response)


if __name__ == "__main__":
    query = input("Enter your request: ")
    perform_task(query)