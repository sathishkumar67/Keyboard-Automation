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
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def perform_task(query: str):
    # System context - only included once at the beginning
    SYSTEM_CONTEXT = {
        "role": "system",
        "content": """
            You are an expert GUI automation agent that performs tasks through keyboard interactions. You analyze screenshots and user queries to determine the exact sequence of keyboard actions(only keyboard shortcuts or typing and not pressing or clicking buttons) needed.\n\nProcess Flow:\n1. Analyze the current screenshot and query to understand the task\n2. Reason step-by-step about required keyboard actions\n3. Execute using PyAutoGUI when keyboard actions are needed\n\nResponse Format:\n1. For reasoning steps: Output plain text describing your thought process\n2. For keyboard actions: Output ONLY a JSON object with this exact structure\n{\n  \"tool\": \"action\",\n  \"description\": \"Detailed description including target location (e.g., address bar, search field)\",\n  \"program\": \"Raw Python script using only PyAutoGUI\"\n}\n\nKeyboard Action Guidelines:\n1. Always use keyboard shortcuts for navigation when possible (Tab, Alt+Tab, Ctrl+L, etc.)\n2. Specify exact target locations in descriptions\n3. PyAutoGUI scripts must be self-contained and use only PyAutoGUI\n4. Include necessary delays and focus actions before typing\n\n\nExample Usage:\nUser query: \"Search for weather in New York\"\nYour response:\nI need to first focus the browser's address bar, then type the search query.\n{\"tool\": \"action\", \"description\": \"Focus address bar using Ctrl+L shortcut\", \"program\": \"import pyautogui\\npyautogui.hotkey('ctrl', 'l')\"}\n{\"tool\": \"action\", \"description\": \"Type search query in address bar\", \"program\": \"import pyautogui\\npyautogui.write('weather New York')\\npyautogui.press('enter')\"}
            For now don't do any reasoning steps, just give me the JSON object with the first action you need to take and wait feedback by a new screenshot before continuing.
            After completing the objective that the user has given you, always respond with:
            {"tool": "task_complete"}
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
    messages.append(initial_user_message)
    
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
        
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=current_messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None
        )

        response = completion.choices[0].message.content
        print(response)
        
        # Add the AI's response to the conversation history
        ai_response_message = {
            "role": "assistant",
            "content": response
        }
        messages.append(ai_response_message)

        # Extract the dictionary part
        start = response.find('{')
        end = response.rfind('}') + 1
        
        if start != -1 and end != 0:
            dict_str = response[start:end]
            try:
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
                    print("Task completed successfully.")
                    break
                else:
                    print("Unknown tool in response:", parsed.get("tool"))
                    break
            except Exception as e:  
                print("Error parsing or executing response:", e)
                # Add error feedback to context
                error_feedback = {
                    "role": "user", 
                    "content": f"Error occurred: {str(e)}. Please adjust your approach."
                }
                messages.append(error_feedback)
        else:
            print("No JSON object found in response:", response)
            # Add feedback about invalid response format
            format_feedback = {
                "role": "user",
                "content": "Your response should contain a JSON object with the required structure. Please provide a valid response."
            }
            messages.append(format_feedback)

if __name__ == "__main__":
    query = input("Enter your request: ")
    perform_task(query)