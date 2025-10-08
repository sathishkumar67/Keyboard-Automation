from __future__ import annotations
import os
import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from groq import Groq
from dotenv import load_dotenv
from capture import take_screenshot
from utils import encode_image_to_data_uri

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# System prompts for master-slave architecture
MASTER_SYSTEM_PROMPT = """
You are the MASTER agent in a GUI keyboard automation system. 
You behave like a structured planner (like Cursor), with these responsibilities:
1. Read the user's query and current screen state (screenshot).
2. Break down the high-level task into clear, atomic keyboard-based actions.
3. Output structured decisions in JSON — no natural language explanations.

Each response must use one of the following JSON schemas exactly:

1. **Planning multiple steps**
{
  "role": "master",
  "type": "plan",
  "steps": ["step 1", "step 2", ...],
  "next_action": "the immediate next action to delegate"
}

2. **Delegating a single atomic action**
{
  "role": "master",
  "type": "delegate",
  "instruction": "precise executor instruction",
  "expected_result": "what should happen after execution"
}

3. **Task completion**
{
  "role": "master",
  "type": "complete",
  "reason": "short reason why task is complete"
}

4. **Error handling**
{
  "role": "master",
  "type": "error",
  "issue": "what went wrong",
  "recovery": "short recovery plan"
}

Rules:
- Always produce deterministic JSON — no text before or after.
- Use short, imperative phrasing in 'instruction' (e.g., "Focus address bar", "Open new tab").
- Prefer stepwise plans for multi-step tasks.
- Use delegate type when you can issue a concrete single keyboard action.
- Do not explain your reasoning outside JSON.
"""


SLAVE_SYSTEM_PROMPT = """
You are the SLAVE agent — a precise keyboard action executor, like a Cursor tool.
You receive a single focused instruction and current screen state.
You must translate the instruction into deterministic, executable Python code using ONLY pyautogui for keyboard actions.

**Constraints**:
- Always import pyautogui.
- Use only: hotkey(), write(), press() — no mouse, no clicks.
- Be exact and minimal — 1-3 lines of code typically.
- Never explain your answer. Output only valid JSON in this format:

{
  "role": "slave",
  "action": "short description of what the code does",
  "program": "python code as a single string",
  "confidence": "high" | "medium" | "low"
}

**Examples**:

✅ Example 1:
{
  "role": "slave",
  "action": "Open new browser tab",
  "program": "import pyautogui; pyautogui.hotkey('ctrl', 't')",
  "confidence": "high"
}

✅ Example 2:
{
  "role": "slave",
  "action": "Focus address bar and type URL",
  "program": "import pyautogui; pyautogui.hotkey('ctrl', 'l'); pyautogui.write('https://example.com'); pyautogui.press('enter')",
  "confidence": "high"
}

Rules:
- No natural language outside JSON.
- Be deterministic: avoid guessing.
- Use 'medium' or 'low' confidence only when the instruction is ambiguous.
"""

class MasterSlaveGUIAgent:
    def __init__(self):
        self.client = client
        self.master_history: List[Dict] = []
        self.slave_history: List[Dict] = []
        self.current_plan: List[str] = []
        self.current_step: int = 0
        self.retry_count: int = 0
        self.max_retries: int = 3
        
    def get_fresh_screenshot(self) -> str:
        """Take and encode a fresh screenshot"""
        return encode_image_to_data_uri(take_screenshot())
    
    def call_model(self, messages: List[Dict], model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> str:
        """Make API call to Groq"""
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,  # Lower temperature for more consistent actions
                max_completion_tokens=1024,
                top_p=0.9,
                stream=False,
                stop=None
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            raise
    
    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from model response"""
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"JSON extraction failed: {e}")
        return None
    
    def master_planning_phase(self, query: str, screenshot: str) -> Dict:
        """Master analyzes task and creates plan"""
        messages = [
            {"role": "system", "content": MASTER_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"User Task: {query}"},
                {"type": "image_url", "image_url": {"url": screenshot}}
            ]}
        ]
        
        response = self.call_model(messages)
        print(f"Master Analysis: {response}")
        
        master_decision = self.extract_json(response)
        
        if not master_decision:
            # Fallback to direct delegation
            master_decision = {
                "role": "master", 
                "type": "delegate", 
                "instruction": query,
                "expected_result": "Complete the requested task"
            }
        
        self.master_history.append(master_decision)
        return master_decision
    
    def slave_execution_phase(self, instruction: str, screenshot: str) -> Dict:
        """Slave executes specific instruction"""
        messages = [
            {"role": "system", "content": SLAVE_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Master Instruction: {instruction}"},
                {"type": "image_url", "image_url": {"url": screenshot}}
            ]}
        ]
        
        response = self.call_model(messages)
        print(f"Slave Action: {response}")
        
        slave_action = self.extract_json(response)
        
        if not slave_action:
            # Fallback safe action
            slave_action = {
                "role": "slave",
                "action": "Safe fallback - Press Escape",
                "program": "import pyautogui; pyautogui.press('esc')",
                "confidence": "low"
            }
        
        self.slave_history.append(slave_action)
        return slave_action
    
    def execute_action_safely(self, program: str) -> Tuple[bool, str]:
        """Safely execute pyautogui program with error handling"""
        try:
            # Ensure import is present and safe execution
            safe_program = program
            if "import pyautogui" not in safe_program:
                safe_program = "import pyautogui\n" + safe_program
            
            # Add small delays between actions for stability
            safe_program = safe_program.replace(';', '; ')
            
            # Execute in a controlled environment
            exec_globals = {"pyautogui": __import__("pyautogui")}
            exec(safe_program, exec_globals)
            
            return True, "Action executed successfully"
        except Exception as e:
            return False, f"Execution failed: {str(e)}"
    
    def master_verification_phase(self, query: str, screenshot: str, executed_action: Dict) -> Dict:
        """Master verifies if action was successful and plans next step"""
        messages = [
            {"role": "system", "content": MASTER_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Original Task: {query}\nExecuted Action: {json.dumps(executed_action)}\nVerify result and determine next step:"},
                {"type": "image_url", "image_url": {"url": screenshot}}
            ]}
        ]
        
        # Add recent master history for context
        if self.master_history:
            messages.insert(1, {"role": "user", "content": f"Recent plan context: {json.dumps(self.master_history[-1])}"})
        
        response = self.call_model(messages)
        print(f"Master Verification: {response}")
        
        verification = self.extract_json(response)
        
        if not verification:
            verification = {"role": "master", "type": "delegate", "instruction": "Continue with next step"}
        
        self.master_history.append(verification)
        return verification
    
    def optimize_common_tasks(self, query: str) -> Optional[Dict]:
        """Direct optimization for common browser tasks to reduce API calls"""
        lower_query = query.lower()
        
        # Browser navigation patterns
        url_match = re.search(r'(https?://[^\s]+|www\.[^\s]+)', query)
        if url_match and any(keyword in lower_query for keyword in ['open', 'go to', 'navigate', 'visit']):
            url = url_match.group(0)
            return {
                "role": "slave",
                "action": f"Navigate to {url}",
                "program": f"import pyautogui; pyautogui.hotkey('ctrl', 'l'); pyautogui.write('{url}'); pyautogui.press('enter')",
                "confidence": "high"
            }
        
        # Search patterns
        if 'search' in lower_query:
            search_terms = query.replace('search', '').replace('for', '').strip()
            return {
                "role": "slave", 
                "action": f"Search for {search_terms}",
                "program": f"import pyautogui; pyautogui.hotkey('ctrl', 'k'); pyautogui.write('{search_terms}'); pyautogui.press('enter')",
                "confidence": "high"
            }
        
        # Common browser actions
        action_map = {
            'new tab': ("Open new tab", "import pyautogui; pyautogui.hotkey('ctrl', 't')"),
            'close tab': ("Close current tab", "import pyautogui; pyautogui.hotkey('ctrl', 'w')"),
            'refresh': ("Refresh page", "import pyautogui; pyautogui.hotkey('f5')"),
            'back': ("Go back", "import pyautogui; pyautogui.hotkey('alt', 'left')"),
            'forward': ("Go forward", "import pyautogui; pyautogui.hotkey('alt', 'right')"),
        }
        
        for pattern, (action, program) in action_map.items():
            if pattern in lower_query:
                return {
                    "role": "slave",
                    "action": action,
                    "program": program,
                    "confidence": "high"
                }
        
        return None
    
    def perform_task(self, query: str) -> None:
        """Main task execution with master-slave architecture"""
        print(f"Starting task: {query}")
        
        # Check for optimized common tasks first
        optimized_action = self.optimize_common_tasks(query)
        if optimized_action:
            print("Using optimized action for common task")
            success, message = self.execute_action_safely(optimized_action["program"])
            if success:
                print("Optimized task completed successfully!")
                return
            else:
                print(f"Optimized action failed, falling back to master-slave: {message}")
        
        # Initial master planning
        current_screenshot = self.get_fresh_screenshot()
        master_plan = self.master_planning_phase(query, current_screenshot)
        
        # Main execution loop
        while self.retry_count < self.max_retries:
            # Determine current action based on master's decision
            if master_plan.get("type") == "delegate":
                instruction = master_plan.get("instruction", query)
                
                # Slave execution
                slave_action = self.slave_execution_phase(instruction, current_screenshot)
                
                if slave_action.get("role") == "slave" and "program" in slave_action:
                    # Execute the action
                    success, message = self.execute_action_safely(slave_action["program"])
                    print(f"Action execution: {message}")
                    
                    # Wait for action to take effect
                    time.sleep(2)
                    
                    # Get new screenshot for verification
                    new_screenshot = self.get_fresh_screenshot()
                    
                    # Master verification
                    master_plan = self.master_verification_phase(
                        query, new_screenshot, slave_action
                    )
                    
                    current_screenshot = new_screenshot
                    
                    if success:
                        self.retry_count = 0
                    else:
                        self.retry_count += 1
                else:
                    self.retry_count += 1
            
            elif master_plan.get("type") == "complete":
                print(f"Task completed: {master_plan.get('reason', 'No reason provided')}")
                break
            
            elif master_plan.get("type") == "plan":
                # Execute planned steps
                steps = master_plan.get("steps", [])
                next_action = master_plan.get("next_action")
                
                if next_action and steps:
                    print(f"Executing planned step: {next_action}")
                    # Convert plan step to delegation
                    master_plan = {
                        "role": "master",
                        "type": "delegate",
                        "instruction": next_action,
                        "expected_result": f"Complete step: {next_action}"
                    }
                else:
                    print("No actionable steps in plan")
                    break
            
            else:
                print(f"Unknown master decision type: {master_plan.get('type')}")
                self.retry_count += 1
            
            # Safety check
            if len(self.master_history) > 10:
                print("Safety limit reached - stopping execution")
                break
        
        if self.retry_count >= self.max_retries:
            print(f"Task failed after {self.max_retries} retries")
        
        print("Execution completed")


def perform_task(query: str) -> None:
    """Main entry point - maintains original function signature"""
    agent = MasterSlaveGUIAgent()
    agent.perform_task(query)


if __name__ == "__main__":
    user_query = input("Enter your request: ")
    perform_task(user_query)