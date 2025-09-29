from __future__ import annotations
import os
import pyautogui
from gradio_client import Client, handle_file
from capture import take_screenshot
from model_api import split_text
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read Gradio client URL from environment variable
gradio_client_url = os.getenv("GRADIO_CLIENT_URL")

# Initialize Gradio client for model API interaction
client = Client(gradio_client_url)

def perform_task(task_description: str) -> dict:
    """
    Sends a screenshot and task description to the model API and returns the predicted click coordinates.

    Args:
        task_description (str): The description of the GUI task to perform.

    Returns:
        dict: The model's output containing 'x' and 'y' coordinates for the click action.
    """
    result = client.predict(
        image=handle_file(take_screenshot()),
        task=task_description,
        api_name="/predict"
    )
    return result

# Main loop for interactive task input and GUI automation
while True:
    task = input("Enter a task description (or 'exit' to quit): ")
    if task.lower() == 'exit':
        break
    # Split the task into actionable lines if needed
    # lines = split_text(task)
    lines = [line.strip() for line in task.split(",")]
	
    for i in range(len(lines)):
        # Get click coordinates from model API
        output = perform_task(lines[i])
        # Move mouse to predicted coordinates and perform click
        pyautogui.moveTo(output['x'], output['y'], duration=0.25)
        pyautogui.click()