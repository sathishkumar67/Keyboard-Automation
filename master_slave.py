from __future__ import annotations
import os
import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
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

# System prompts for different roles
MASTER_SYSTEM_PROMPT = """
You are a master GUI automation planner. Your role is to:
1. Analyze the user's high-level task and break it down into logical steps
2. Determine when to delegate to the slave executor
3. Monitor progress and adjust plans based on results
4. Declare task completion only when the entire objective is achieved

You receive screenshots and decide whether to:
- Create a multi-step plan for complex tasks
- Delegate immediate actions to the slave
- Verify task completion

Output format:
- For planning: {"role": "planner", "plan": ["step1", "step2", ...], "next_action": "delegate_to_slave"}
- For verification: {"role": "verifier", "status": "complete/incomplete", "reason": "..."}
- For delegation: {"role": "master", "instruction": "specific action for slave", "expectation": "what to expect"}
"""

SLAVE_SYSTEM_PROMPT = """
You are an expert GUI automation executor. Your role is to:
1. Execute single, focused actions based on the master's instructions
2. Analyze screenshots to determine exact keyboard actions needed
3. Report results back to the master
4. Handle retries and error recovery for individual actions

You can only interact with the GUI using keyboard actions. Use keyboard shortcuts to navigate and perform tasks.

Process Flow:
1. Analyze current screenshot and master's instruction
2. Determine the exact keyboard action sequence
3. Execute and report results

Output format:
- For actions: {"role": "executor", "action": "description", "program": "pyautogui code", "confidence": "high/medium/low"}
- For results: {"role": "executor", "result": "success/partial/failure", "observation": "what happened", "suggestion": "next step"}
- For errors: {"role": "executor", "error": "description", "recovery_action": "suggested fix"}
"""

ACTION_VALIDATION_PROMPT = """
Quickly validate if this action makes sense for the current screen. Respond with JSON only:
{"valid": true/false, "reason": "brief explanation", "suggestion": "if invalid"}
"""

class GUIAutomationAgent:
    def __init__(self):
        self.client = client
        self.conversation_history: List[Dict] = []
        self.current_plan: List[str] = []
        self.current_step: int = 0
        self.last_screenshot: Optional[str] = None
        self.retry_count: int = 0
        self.max_retries: int = 3
        
    def get_fresh_screenshot(self) -> str:
        """Take and encode a fresh screenshot"""
        screenshot_path = take_screenshot()
        return encode_image_to_base64(screenshot_path)
    
    def call_model(self, messages: List[Dict], model: str = "gpt-4.1") -> str:
        """Make API call to Azure OpenAI"""
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,  # Lower temperature for more consistent actions
                top_p=0.9,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            raise
    
    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from model response"""
        try:
            # Look for JSON pattern
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return None
    
    def validate_action(self, action: Dict, screenshot: str) -> Tuple[bool, str]:
        """Quick validation of proposed action"""
        validation_messages = [
            {"role": "system", "content": ACTION_VALIDATION_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Action to validate: {json.dumps(action)}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}}
            ]}
        ]
        
        try:
            response = self.call_model(validation_messages)
            validation = self.extract_json(response)
            if validation and validation.get("valid", False):
                return True, validation.get("suggestion", "Action looks good")
            else:
                return False, validation.get("reason", "Action invalid")
        except:
            return True, "Validation skipped"  # Proceed if validation fails
    
    def master_planning_phase(self, query: str, screenshot: str) -> Dict:
        """Master analyzes task and creates initial plan"""
        messages = [
            {"role": "system", "content": MASTER_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"User task: {query}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}}
            ]}
        ]
        
        response = self.call_model(messages)
        master_decision = self.extract_json(response)
        
        if not master_decision:
            # Fallback: delegate directly to slave
            master_decision = {
                "role": "master", 
                "instruction": query,
                "expectation": "Complete the requested task"
            }
        
        return master_decision
    
    def slave_execution_phase(self, instruction: str, screenshot: str, context: List[Dict] = None) -> Dict:
        """Slave executes specific instructions"""
        messages = [
            {"role": "system", "content": SLAVE_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Master instruction: {instruction}\n\nPrevious context: {json.dumps(context[-2:] if context else [])}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}}
            ]}
        ]
        
        response = self.call_model(messages)
        slave_action = self.extract_json(response)
        
        if not slave_action:
            # Fallback action
            slave_action = {
                "role": "executor",
                "action": "Fallback: Press Escape to clear state",
                "program": "import pyautogui; pyautogui.press('esc')",
                "confidence": "low"
            }
        
        return slave_action
    
    def execute_action_program(self, program: str) -> bool:
        """Safely execute pyautogui program"""
        try:
            # Add import if not present
            if "import pyautogui" not in program:
                program = "import pyautogui\n" + program
            
            # Execute the program
            exec(program, {"pyautogui": __import__("pyautogui")})
            return True
        except Exception as e:
            print(f"Action execution failed: {e}")
            return False
    
    def master_verification_phase(self, query: str, screenshot: str, history: List[Dict]) -> Dict:
        """Master verifies if task is complete"""
        recent_history = history[-3:]  # Last 3 interactions
        
        messages = [
            {"role": "system", "content": MASTER_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Original task: {query}\nRecent actions: {json.dumps(recent_history)}\nIs the task complete?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}}
            ]}
        ]
        
        response = self.call_model(messages)
        verification = self.extract_json(response)
        
        if not verification:
            verification = {"role": "verifier", "status": "incomplete", "reason": "Unable to verify"}
        
        return verification
    
    def perform_task(self, query: str) -> None:
        """Main task execution loop with master-slave architecture"""
        print(f"Starting task: {query}")
        
        # Initial screenshot
        current_screenshot = self.get_fresh_screenshot()
        self.last_screenshot = current_screenshot
        
        # Master creates initial plan
        master_plan = self.master_planning_phase(query, current_screenshot)
        self.conversation_history.append({"role": "master", "content": master_plan})
        print(f"Master plan: {master_plan}")
        
        # Check if task is immediately complete (simple tasks)
        if master_plan.get("role") == "verifier" and master_plan.get("status") == "complete":
            print("Task completed in planning phase!")
            return
        
        # Main execution loop
        while self.retry_count < self.max_retries:
            # Get current instruction from master
            current_instruction = master_plan.get("instruction", query)
            
            # Slave execution
            slave_result = self.slave_execution_phase(
                current_instruction, 
                current_screenshot,
                self.conversation_history
            )
            
            print(f"Slave action: {slave_result}")
            
            # Execute action if it's an action type
            if slave_result.get("role") == "executor" and "program" in slave_result:
                # Validate action before execution
                is_valid, reason = self.validate_action(slave_result, current_screenshot)
                
                if is_valid:
                    # Execute the action
                    success = self.execute_action_program(slave_result["program"])
                    
                    # Wait for action to take effect
                    time.sleep(2)
                    
                    # Get new screenshot
                    new_screenshot = self.get_fresh_screenshot()
                    
                    # Record result
                    execution_result = {
                        "role": "executor",
                        "result": "success" if success else "failure",
                        "action": slave_result.get("action", "unknown"),
                        "observation": "Action executed" if success else "Action failed",
                        "screenshot_changed": new_screenshot != current_screenshot
                    }
                    
                    self.conversation_history.append(execution_result)
                    current_screenshot = new_screenshot
                    self.last_screenshot = current_screenshot
                    
                    # Reset retry count on success
                    if success:
                        self.retry_count = 0
                    else:
                        self.retry_count += 1
                else:
                    print(f"Action validation failed: {reason}")
                    self.retry_count += 1
                    
                    # Record validation failure
                    self.conversation_history.append({
                        "role": "executor", 
                        "result": "validation_failed", 
                        "reason": reason
                    })
            else:
                # Handle non-action responses (results, errors, etc.)
                self.conversation_history.append(slave_result)
                
                if slave_result.get("result") in ["success", "partial"]:
                    self.retry_count = 0
                else:
                    self.retry_count += 1
            
            # Master verification
            verification = self.master_verification_phase(
                query, current_screenshot, self.conversation_history
            )
            
            print(f"Master verification: {verification}")
            
            # Check for completion
            if (verification.get("role") == "verifier" and 
                verification.get("status") == "complete"):
                print("Task completed successfully!")
                break
            
            # Update master plan for next iteration if needed
            if (verification.get("status") == "incomplete" and 
                "next_instruction" in verification):
                master_plan = {
                    "role": "master",
                    "instruction": verification["next_instruction"],
                    "expectation": verification.get("reason", "Continue task")
                }
            
            # Safety check - prevent infinite loops
            if len(self.conversation_history) > 20:
                print("Safety limit reached - stopping execution")
                break
        
        if self.retry_count >= self.max_retries:
            print(f"Task failed after {self.max_retries} retries")
        
        print("Task execution completed")


class OptimizedGUIAgent:
    """Even more optimized version for common tasks"""
    
    def __init__(self):
        self.agent = GUIAutomationAgent()
        self.common_patterns = {
            "browser_navigation": ["ctrl+l", "type_url", "enter"],
            "search": ["ctrl+k", "type_query", "enter"],
            "new_tab": ["ctrl+t"],
            "close_tab": ["ctrl+w"],
            "refresh": ["f5"]
        }
    
    def perform_task_optimized(self, query: str) -> None:
        """Optimized task performance with pattern recognition"""
        # Check for common patterns to avoid unnecessary planning
        lower_query = query.lower()
        
        # Direct pattern matching for common tasks
        if any(pattern in lower_query for pattern in ["open website", "go to", "navigate to"]):
            print("Using optimized browser navigation pattern")
            self._handle_browser_navigation(query)
        elif "search" in lower_query:
            print("Using optimized search pattern")
            self._handle_search(query)
        else:
            # Fall back to full master-slave architecture
            self.agent.perform_task(query)
    
    def _handle_browser_navigation(self, query: str):
        """Optimized handler for browser navigation"""
        # Extract URL from query
        url_match = re.search(r'(https?://[^\s]+|www\.[^\s]+)', query)
        if url_match:
            url = url_match.group(0)
            # Direct execution without master planning
            program = f"""
import pyautogui
pyautogui.hotkey('ctrl', 'l')  # Focus address bar
pyautogui.write('{url}')
pyautogui.press('enter')
"""
            self.agent.execute_action_program(program)
            time.sleep(3)  # Wait for page load
        else:
            # Fall back to full architecture
            self.agent.perform_task(query)
    
    def _handle_search(self, query: str):
        """Optimized handler for search tasks"""
        # Extract search query
        search_terms = query.replace("search for", "").replace("search", "").strip()
        
        program = f"""
import pyautogui
pyautogui.hotkey('ctrl', 'k')  # Focus search box (common shortcut)
pyautogui.write('{search_terms}')
pyautogui.press('enter')
"""
        self.agent.execute_action_program(program)
        time.sleep(2)


def perform_task(query: str, optimized: bool = True) -> None:
    """Main entry point with optimization option"""
    if optimized:
        agent = OptimizedGUIAgent()
        agent.perform_task_optimized(query)
    else:
        agent = GUIAutomationAgent()
        agent.perform_task(query)


if __name__ == "__main__":
    user_query = input("Enter your request: ")
    perform_task(user_query, optimized=True)