# necessary imports
from __future__ import annotations
import os
import json
from groq import Groq
from dotenv import load_dotenv
from capture import take_screenshot
from utils import encode_image_to_data_uri

# Load environment variables from the .env file
load_dotenv()

# Initialize the Groq client with the API key from environment variables
client = Groq(api_key=os.getenv("GROQ_API_KEY1"))

# Main function to perform the task based on user query
def perform_task(query: str) -> None:
    # System context - only included once at the beginning
    SYSTEM_CONTEXT = {
        "role": "system",
        "content": """
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
    }
    
    # Initialize messages with system context
    messages = [SYSTEM_CONTEXT]
    
    # Add the initial user query
    initial_user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": query
            }
        ]
    }
    
    # Add the initial user message to messages
    messages.append(initial_user_message)
    
    # Main interaction loop
    while True:
        # Take current screenshot
        screen_shot = encode_image_to_data_uri(take_screenshot())
        
        # Create current state message with screenshot
        current_state_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Current screen state:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": screen_shot
                    }
                }
            ]
        }
        
        # Add current state to messages (maintaining full context)
        current_messages = messages + [current_state_message]
        
        # Get completion from the model
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=current_messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None
        )

        # Extract the model's response
        response = completion.choices[0].message.content
        print(response)
        
        # Add the AI's response to the conversation history
        ai_response_message = {
            "role": "assistant",
            "content": response
        }
        # Append the AI response to messages
        messages.append(ai_response_message)

        # Extract the dictionary part
        start = response.find('{')
        end = response.rfind('}') + 1
        
        # If a JSON object is found, parse and execute the action
        if start != -1 and end != 0:
            # Extract the JSON string
            dict_str = response[start:end]
            try:
                # Parse the JSON string to a dictionary
                parsed = json.loads(dict_str)
                if parsed.get("tool") == "action":
                    # Execute the action
                    exec(parsed.get("program"))
                    # Add a user message indicating the action was executed
                    action_feedback = {
                        "role": "user",
                        "content": f"Action executed: {parsed.get('description')}. Please provide the next action based on the new screen state."
                    }
                    messages.append(action_feedback)
                elif parsed.get("tool") == "task_complete":
                    # Task is complete, exit the loop
                    print("Task completed successfully.")
                    break
                else:
                    # Unknown tool, provide feedback
                    print("Unknown tool in response:", parsed.get("tool"))
                    break
            except Exception as e: 
                # Handle JSON parsing or execution errors 
                print("Error parsing or executing response:", e)
                # Add error feedback to context
                error_feedback = {
                    "role": "user", 
                    "content": f"Error occurred: {str(e)}. Please adjust your approach."
                }
                # Append error feedback to messages
                messages.append(error_feedback)
        else:
            # No JSON object found, provide feedback
            print("No JSON object found in response:", response)
            # Add feedback about invalid response format
            format_feedback = {
                "role": "user",
                "content": "Your response should contain a JSON object with the required structure. Please provide a valid response."
            }
            # Append format feedback to messages
            messages.append(format_feedback)

if __name__ == "__main__":
    # Enter the user's request
    query = input("Enter your request: ")
    # Call the function to perform the task
    perform_task(query)