from __future__ import annotations
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read API key from environment variable
api_key = os.getenv("GROQ_API_KEY")


def split_text(user_input) -> list[str]:
    """
    Splits the input text into multiple lines at each comma (,) or the word "and".
    The function uses the Groq API with a system prompt to format the text.
    It removes the comma or "and" and preserves the order of words.
    No explanation or extra text is added; only the formatted output is returned.

    Args:
        user_input (str): The input string to be split.

    Returns:
        list[str]: A list of strings, each representing a line after splitting.
    """
    # Initialize Groq client with API key
    client = Groq(api_key=api_key)
    
    # Create a chat completion request with formatting instructions
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are a text formatter. 
                            Your task is to split the user’s input into multiple lines whenever there is a comma (,) or the word "and". 
                            Do not provide any explanation or additional text—only return the formatted output. 
                            Preserve the exact order of words from the input without altering them. 
                            The only transformation you should apply is inserting line breaks at each comma or "and" and remove the "and" or ","
                            """
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0.6,
        max_completion_tokens=1024,
        top_p=0.95,
        stream=True,
        stop=None
    )

    # Collect the streamed response chunks into a single string
    full_response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content

    # Split the response into lines and return as a list of strings
    lines = full_response.strip().split("\n")
    return lines