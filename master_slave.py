from __future__ import annotations
import os
import json
import time
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
    azure_endpoint="https://harsh-mamhtiwt-eastus2.cognitiveservices.azure.com/  "
)

# Master planner prompt - focuses on high-level strategy and task decomposition
MASTER_PROMPT = """
You are an expert GUI automation task planner. Your role is to:
1. Analyze the user query and current screen state
2. Break down the complex task into smaller, executable sub-tasks
3. Plan the sequence of actions needed to complete the overall task
4. Monitor progress and adjust the plan as needed

You receive the current screenshot and user query, then output a high-level plan with:
- Current task status
- Next sub-task to execute
- Success criteria for each sub-task
- Plan adjustments if previous actions failed

Output Format:
{
    "tool": "planning",
    "current_task": "Description of the current sub-task",
    "success_criteria": "How to verify this sub-task was completed successfully",
    "next_action_type": "action/verification/wait",
    "plan_status": "planning/in_progress/completed/failed",
    "reasoning": "Your step-by-step reasoning about the current state and next steps"
}

If the overall task is complete, output:
{
    "tool": "task_complete",
    "summary": "Brief summary of what was accomplished",
    "reasoning": "Why the task is considered complete"
}
"""

# Slave executor prompt - focuses on specific action execution
SLAVE_PROMPT = """
You are an expert GUI automation action executor. Your role is to:
1. Execute precise keyboard actions based on the current screen state
2. Use keyboard shortcuts and typing operations
3. You can use any Python library for keyboard actions including pyautogui, keyboard, pynput, or others
4. Provide exact code to perform the requested action
5. Verify the action was successful

Available libraries and their usage:
- keyboard: keyboard.send('ctrl+v'), keyboard.write('text'), keyboard.send('ctrl+t'), keyboard.send('ctrl+l')
- pynput: from pynput.keyboard import Key, Controller; keyboard = Controller(); keyboard.press(Key.ctrl); keyboard.press('v'); keyboard.release('v'); keyboard.release(Key.ctrl)
- pyautogui: pyautogui.hotkey('ctrl', 't'), pyautogui.hotkey('ctrl', 'l'), pyautogui.write('text'), pyautogui.press('enter'), pyautogui.hotkey('ctrl', 'v'), pyautogui.hotkey('ctrl', 'c'), pyautogui.hotkey('ctrl', 'x'), pyautogui.hotkey('tab'), pyautogui.hotkey('shift', 'tab')

When navigating or focusing on elements, always use appropriate keyboard shortcuts like:
- Ctrl+L to focus address bar
- Ctrl+T to open new tab
- Ctrl+W to close tab
- Ctrl+Tab to switch tabs
- Ctrl+Shift+Tab to switch tabs backward
- Tab/Shift+Tab to navigate elements
- Ctrl+C/Ctrl+V for copy/paste

Output Format:
{
    "tool": "action",
    "description": "Clear description of what the action does",
    "program": "Python code to execute the action using appropriate libraries",
    "expected_result": "What should happen after this action",
    "verification_hint": "How to verify the action was successful in the next screenshot"
}

Example:
{
    "tool": "action",
    "description": "Open new tab and navigate to Google",
    "program": "import keyboard; keyboard.send('ctrl+t'); time.sleep(0.5); keyboard.send('ctrl+l'); keyboard.write('https://www.google.com'); keyboard.send('enter')",
    "expected_result": "New tab opens with Google homepage",
    "verification_hint": "Check for Google logo and search bar"
}
"""


class MasterSlaveGUIAgent:
    def __init__(self):
        self.client = client
        self.messages_history = []
        self.task_completed = False
        self.max_iterations = 50  # Prevent infinite loops
        self.iteration_count = 0
        
    def call_model(self, messages, model_name="gpt-4.1"):
        """Make API call to Azure OpenAI with error handling"""
        try:
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,  # Lower temperature for more consistent planning
                top_p=0.9,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling Azure OpenAI: {e}")
            return f"API call failed: {str(e)}. Please retry or adjust."

    def extract_json_from_response(self, response_text):
        """Extract JSON object from response text"""
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = response_text[start:end]
            try:
                return json.loads(json_str)
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                return None
        return None

    def get_current_screenshot_data(self):
        """Take screenshot and encode it"""
        screenshot_path = take_screenshot()
        image_base64 = encode_image_to_base64(screenshot_path)
        return image_base64

    def execute_action(self, action_json):
        """Execute the action and handle errors"""
        try:
            program_code = action_json.get("program", "")
            if program_code:
                exec(program_code)
                return True, "Action executed successfully"
        except Exception as e:
            error_msg = f"Action execution failed: {str(e)}"
            print(error_msg)
            return False, error_msg
        return False, "No program code to execute"

    def run_master_planning(self, user_query, current_image_base64):
        """Master: Plan the high-level task"""
        messages = [
            {"role": "system", "content": MASTER_PROMPT},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": f"User Query: {user_query}\n\nCurrent Task: Plan the automation steps needed to complete this task. Consider the current screen state and break down into executable sub-tasks."},
                    {"type": "image_url", "image_url": {"url": f"image/jpeg;base64,{current_image_base64}"}}
                ]
            }
        ]
        
        response = self.call_model(messages)
        print(f"Master Planning: {response}")
        
        parsed_response = self.extract_json_from_response(response)
        if parsed_response:
            return parsed_response
        else:
            # Fallback: create a simple action request
            return {
                "tool": "planning",
                "current_task": "Continue with user query",
                "success_criteria": "Progress toward user goal",
                "next_action_type": "action",
                "plan_status": "in_progress",
                "reasoning": "Failed to parse master response, continuing with basic approach"
            }

    def run_slave_execution(self, planning_info, current_image_base64):
        """Slave: Execute specific actions based on planning"""
        task_description = planning_info.get("current_task", "Perform user request")
        reasoning = planning_info.get("reasoning", "Execute the next step")
        
        messages = [
            {"role": "system", "content": SLAVE_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Current Task: {task_description}\n\nReasoning: {reasoning}\n\nBased on the current screen state, provide the specific keyboard action to execute this task. Focus on precise, executable code using keyboard shortcuts for navigation."},
                    {"type": "image_url", "image_url": {"url": f"image/jpeg;base64,{current_image_base64}"}}
                ]
            }
        ]
        
        response = self.call_model(messages)
        print(f"Slave Execution: {response}")
        
        parsed_response = self.extract_json_from_response(response)
        if parsed_response:
            return parsed_response
        else:
            # Fallback: indicate need for re-planning
            return {
                "tool": "action",
                "description": "Need to re-plan due to parsing error",
                "program": "",
                "expected_result": "Re-plan the approach",
                "verification_hint": "Wait for new plan"
            }

    def verify_action_success(self, action_info, current_image_base64):
        """Verify if the previous action was successful"""
        expected_result = action_info.get("expected_result", "Action completed")
        verification_hint = action_info.get("verification_hint", "Check screenshot for changes")
        
        # Instead of calling API again, we'll use the next screenshot automatically
        # This is handled in the main loop by checking the new state
        return True  # Assume success, let next iteration verify

    def run(self, user_query):
        """Main execution loop with master-slave architecture"""
        print(f"Starting task: {user_query}")
        
        # Initial screenshot
        current_image_base64 = self.get_current_screenshot_data()
        
        # Initial planning by master
        planning_info = self.run_master_planning(user_query, current_image_base64)
        
        while not self.task_completed and self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            print(f"\n--- Iteration {self.iteration_count} ---")
            
            # Check if task is complete
            if planning_info.get("tool") == "task_complete":
                print("Task completed successfully by master planner.")
                print(planning_info.get("summary", "Task completed."))
                self.task_completed = True
                break
            
            # Slave execution phase
            if planning_info.get("next_action_type") == "action":
                action_info = self.run_slave_execution(planning_info, current_image_base64)
                
                if action_info.get("tool") == "action":
                    # Execute the action
                    success, message = self.execute_action(action_info)
                    
                    if success:
                        print(f"Action executed: {action_info.get('description')}")
                        
                        # Wait briefly for screen to update
                        time.sleep(1)
                        
                        # Get new screenshot for verification
                        current_image_base64 = self.get_current_screenshot_data()
                        
                        # Master re-planning based on new state
                        planning_info = self.run_master_planning(user_query, current_image_base64)
                    else:
                        print(f"Action failed: {message}")
                        # Re-plan based on failure
                        current_image_base64 = self.get_current_screenshot_data()
                        planning_info = self.run_master_planning(user_query, current_image_base64)
                else:
                    print("Invalid action format, re-planning...")
                    current_image_base64 = self.get_current_screenshot_data()
                    planning_info = self.run_master_planning(user_query, current_image_base64)
            else:
                # Non-action planning, just update with new state
                current_image_base64 = self.get_current_screenshot_data()
                planning_info = self.run_master_planning(user_query, current_image_base64)
        
        if not self.task_completed:
            print(f"Task terminated after {self.max_iterations} iterations. Consider the task failed or needing manual intervention.")


def perform_task(query: str) -> None:
    """Main function to run the master-slave GUI agent"""
    agent = MasterSlaveGUIAgent()
    agent.run(query)


if __name__ == "__main__":
    user_query = input("Enter your request: ")
    perform_task(user_query)