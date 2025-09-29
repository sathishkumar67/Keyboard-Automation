from __future__ import annotations
from programmer import get_code
from capture import take_screenshot

# Prompt the user to enter a request or task description
query = input("Enter your request: ")

# Capture a screenshot and get the file path
path = take_screenshot()

# Generate Python code based on the query and screenshot
code = get_code(query, path)
print("Generated Code:\n", code)

try:
    # Attempt to execute the generated code
    exec(code)
except Exception as e:
    # Handle errors during execution of the generated code
    print(f"Error executing generated code: {e}")
    try:
        # Attempt to clean up the code by removing the first and last lines
        code = "\n".join(code.split("\n")[1:-1])
        exec(code)
    except Exception as inner_e:
        # Handle errors during re-execution of the cleaned code
        print(f"Error re-executing cleaned code: {inner_e}")
finally:
    # Indicate that the execution attempt has completed
    print("Execution attempt completed.")