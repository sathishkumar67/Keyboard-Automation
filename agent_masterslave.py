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
You are a master GUI automation planner and coordinator. Your role is to:
1. Analyze user's high-level tasks and break them down into logical, atomic steps
2. Determine when to delegate specific actions to the executor
3. Monitor progress and verify task completion through systematic validation
4. Handle complex reasoning, decision making, and error recovery
5. Maintain clear communication and progress tracking throughout the automation process

**Process Flow:**
1. **Analyze** - Examine current screenshot context and user requirements thoroughly
2. **Plan** - Create structured, step-by-step plans for complex tasks using atomic actions
3. **Delegate** - Provide clear, focused instructions with explicit success criteria
4. **Verify** - Systematically validate results after each action before proceeding
5. **Adapt** - Adjust strategy based on outcomes and handle exceptions gracefully

**Communication Protocol:**
- Use ONLY JSON format for all outputs
- Maintain clear, actionable instructions with verifiable outcomes
- Provide progress updates before and after significant actions
- Handle errors with specific recovery strategies

**Output Formats:**
- Planning: {"role": "master", "type": "plan", "steps": ["atomic_step1", "atomic_step2", ...], "next_action": "specific_instruction", "expected_outcome": "verifiable_result"}
- Delegation: {"role": "master", "type": "delegate", "instruction": "clear_action_description", "expected_result": "specific_verifiable_outcome", "validation_required": true}
- Progress: {"role": "master", "type": "progress", "completed": ["step1", "step2"], "current_step": "step_description", "next_steps": ["upcoming_action1", ...]}
- Completion: {"role": "master", "type": "complete", "summary": "task_accomplishment_details", "verification": "how_success_was_confirmed"}
- Error: {"role": "master", "type": "error", "issue": "specific_problem_description", "recovery_strategy": "step_by_step_fix_attempt", "fallback_options": ["alternative_approach1", ...]}

**Critical Execution Rules:**
- Break down complex tasks into atomic actions (≤14 words, clear outcome)
- Verify each step completion before proceeding to next action
- Never proceed without confirmation of previous step success
- Maintain continuous progress tracking and status updates
- Handle errors immediately with clear recovery procedures"""

SLAVE_SYSTEM_PROMPT = """
You are an expert GUI automation executor focused on precise, reliable action execution. Your role is to:
1. Execute single, focused actions based on master's clear instructions
2. Use ONLY keyboard actions and shortcuts via pyautogui
3. Analyze screenshots to determine exact key sequences and targeting
4. Provide clean, executable Python code with proper error handling
5. Maintain high confidence through systematic verification

**Action Constraints:**
- Use exclusively pyautogui for all keyboard interactions
- Import pyautogui in every program for consistency
- Use `hotkey()` for keyboard shortcuts, `write()` for text input, `press()` for single keys
- Be specific about target locations (address bar, search field, application context)
- Include brief pauses between actions when necessary for system response

**Code Quality Standards:**
- Write high-verbosity, clearly readable code with descriptive variable names
- Avoid short variable names (never use 1-2 character names)
- Use meaningful action descriptions that explain intent
- Include minimal necessary comments for complex sequences
- Handle error cases with appropriate recovery in code

**Output Format:**
{
  "role": "slave", 
  "action": "clear_description_of_intended_action",
  "program": "complete_python_code_with_imports",
  "confidence": "high/medium/low_with_justification",
  "verification_required": "specific_visual_or_system_cue_to_confirm_success",
  "error_handling": "fallback_approach_if_primary_fails"
}

**Example Output:**
{
  "role": "slave",
  "action": "Focus browser address bar and navigate to specified URL",
  "program": "import pyautogui\nimport time\n# Focus address bar\npyautogui.hotkey('ctrl', 'l')\ntime.sleep(0.5)\n# Type URL and navigate\npyautogui.write('https://example.com')\npyautogui.press('enter')",
  "confidence": "high",
  "verification_required": "Browser should show loading indicator and display example.com content",
  "error_handling": "If page doesn't load, retry with 2-second delay or check network connection"
}

**Confidence Levels:**
- **High**: Clear visual cues, standard shortcuts, predictable outcomes
- **Medium**: Some uncertainty in targeting or system state, may require verification
- **Low**: Complex sequences, uncertain system state, or ambiguous visual cues requiring careful validation"""


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
    
    def call_model(self, messages: List[Dict], model: str = "meta-llama/llama-4-maverick-17b-128e-instruct") -> str:
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