from programmer import get_code
query = input("Enter your request: ")
code = get_code(query)
code = "\n".join(code.split("\n")[1:-1])
print("Generated Code:\n", code)
# put this in try except block to catch any errors
try:
    exec(code)
except Exception as e:
    print(f"Error executing generated code: {e}")