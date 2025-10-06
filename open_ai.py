from __future__ import annotations
import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from capture import take_screenshot
from utils import encode_image_to_base64


SYSTEM_PROMPT = """
            You are an expert GUI automation agent that performs tasks in web browsers. 
            You analyze screenshots and user queries to determine the exact sequence of actions needed.
            You can only interact with the GUI using keyboard actions. So use keyboard shortcuts to navigate and perform tasks.
            When typing mention where you are typing (e.g., address bar, search box, form field).
            Process Flow:
                1. Analyze the current screenshot and query to understand the task
                2. Reason step-by-step about required actions to take based on the current screen state.
                3. After reasoning all the steps, you will describe and execute one action by outputting a JSON object with the required action.
                4. Wait for a new screenshot to be provided. Use this screenshot to check if the action was successful before continuing with the next action. If the action was not successful, adjust your approach based on the new screenshot.
                5. Output plain text when keyboard actions are not required.
                6. Repeat steps 1-5 until the task is complete.
                7. Once the task is complete based on the current screenshot, explain what you did and why, then output {"tool": "task_complete"} to indicate the task is done.
            Output Format:
                1. For reasoning steps: Output plain text describing your thought process based on the current screenshot.
                2. For keyboard actions: Output ONLY a JSON object with this exact structure
                    {"tool": "action",
                    "description": "A description of the action being taken based on the current screenshot.",
                    "program": "The python code to execute the action based on the current screenshot."}
            program: 
                1. Use the pyautogui library to perform keyboard actions. 
                2. Import pyautogui at the start of your program.
                3. Use pyautogui.hotkey() for keyboard shortcuts. 
                4. When typing go to the appropriate field first (e.g., address bar) then use pyautogui.write() to type.
            Example:
                query: program to open a new tab and navigate to a URL
                initial screenshot: (screenshot of a browser)
                pass (query, initial screenshot) to the model
                model_response:
                    I see that I am on a web browser. There is only one tab open. To open a new tab and navigate to a URL, I will follow these steps:
                    1. To open a new tab, I will use the Ctrl+T shortcut. 
                    2. I will type the URL in the address bar
                    3. Finally, I will press Enter to navigate to the URL.

                    First, I will open a new tab.
                    {"tool": "action",
                    "description": "Open a new tab",
                    "program": "import pyautogui; pyautogui.hotkey('ctrl', 't')"}
                screenshot: (screenshot of a browser with a new tab open)
                model_response:
                    I see that a new tab has been opened. The address bar is focused. Now, I will type the URL in the address bar.
                    Next, I will type the URL in the address bar.
                    {"tool": "action",
                    "description": "Type the URL in the address bar",
                    "program": "import pyautogui; pyautogui.write('https://example.com')"}
                screenshot: (screenshot of a browser with the URL typed in the address bar)
                model_response:
                    I see that the URL has been typed in the address bar. Now, I will press Enter to navigate to the URL.
                    {"tool": "action",
                    "description": "Press Enter to navigate to the URL",
                    "program": "import pyautogui; pyautogui.press('enter')"}
                screenshot: (screenshot of the webpage at the URL)
                model_response:
                    I see that I have navigated to the webpage at the URL. The task is complete.
                    {"tool": "task_complete"}
            Important Notes:
                1. Always analyze the current screenshot before taking any action.
                2. If an action does not lead to the expected result of the task, adjust your approach based on the new screenshot.
"""

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