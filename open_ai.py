from __future__ import annotations
import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from capture import take_screenshot
from utils import encode_image_to_base64


# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_version="2025-03-01-preview",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint="https://harsh-mamhtiwt-eastus2.cognitiveservices.azure.com/"
)

# System prompt (same as in your original)
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
Important Notes:
    1. Always analyze the current screenshot before taking any action.
    2. If an action does not lead to the expected result of the task, adjust your approach based on the new screenshot.
    3. You will reason step-by-step before taking any action.
Example: # if error occurs remove this
    query: program to open a new tab and navigate to a URL
    initial screenshot: (screenshot of a browser)
    reasoning: I need to open a new tab and navigate to a URL. To do this, I will:
        1. Use the Ctrl+T shortcut to open a new tab.
        2. Use the Ctrl+L shortcut to navigate to the URL.
    action: {"tool": "action", "description": "Open a new tab and navigate to a URL.", "program": "pyautogui.hotkey('ctrl', 't'); pyautogui.hotkey('ctrl', 'l');"}
    """


def perform_task(query: str) -> None:
    # Initialize conversation with system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Add initial user query
    messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(take_screenshot())}"}
                }]
        })

    while True:
        # Take a fresh screenshot
        screenshot_path = take_screenshot()
        image_base64 = encode_image_to_base64(screenshot_path)

        # Append current screen state to messages
        current_state = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Current screen state:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        }

        # Build full message list for this turn
        current_messages = messages + [current_state]

        # Call Azure OpenAI model
        try:
            completion = client.chat.completions.create(
                model="gpt-4.1",  # Replace with your actual deployment name
                messages=current_messages,
                max_tokens=1024,
                temperature=1,
                top_p=1,
                stream=False
            )
        except Exception as e:
            print(f"Error calling Azure OpenAI: {e}")
            messages.append({
                "role": "user",
                "content": f"API call failed: {str(e)}. Please retry or adjust."
            })
            continue

        response_text = completion.choices[0].message.content
        print(response_text)

        # Save assistant response to conversation history
        messages.append({
            "role": "assistant",
            "content": response_text
        })

        # Try to extract JSON object from response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1

        if start != -1 and end > start:
            json_str = response_text[start:end]
            try:
                parsed = json.loads(json_str)
                tool = parsed.get("tool")

                if tool == "action":
                    # Execute the provided Python code
                    exec(parsed.get("program", ""))
                    # Add feedback that action was executed
                    messages.append({
                        "role": "user",
                        "content": f"Action executed: {parsed.get('description')}. Please provide the next action based on the new screen state."
                    })

                elif tool == "task_complete":
                    print("Task completed successfully.")
                    break

                else:
                    print(f"Unknown tool: {tool}")
                    messages.append({
                        "role": "user",
                        "content": f"Unknown tool '{tool}' in response. Please follow the specified output format."
                    })

            except Exception as e:
                print(f"Error parsing or executing action: {e}")
                messages.append({
                    "role": "user",
                    "content": f"Error occurred during execution: {str(e)}. Please adjust your approach."
                })

        else:
            print("No valid JSON object found in response.")
            messages.append({
                "role": "user",
                "content": "Your response must include a valid JSON object with 'tool' field. Please correct your output format."
            })


if __name__ == "__main__":
    user_query = input("Enter your request: ")
    perform_task(user_query)