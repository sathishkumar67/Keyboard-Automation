import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
def get_code(query: str) -> str:
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
        {
            "role": "system",
            "content": "You are an assistant whose sole job is to return a runnable Python script and nothing else. For every user request:\n1. Output only Python source code (no surrounding text, no comments, no explanations, no backticks, no metadata).\n2. The code must be valid Python and executable as-is (include necessary imports, functions, and a runnable entry point if appropriate).\n3. Avoid unnecessary verbosity in the code; prefer clear, minimal, runnable implementations.\n4. If the request cannot be fulfilled safely or would violate policy, briefly return a single-line Python comment explaining the refusal (still no extra prose).\n5. You only use the PyAutoGUI library, always import it at the start of the program and for now you will give only keyboard actions"
        },
            {
                "role": "user",
                "content": f"Write a Python script to {query}"
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None
    )
    full_response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
    return full_response