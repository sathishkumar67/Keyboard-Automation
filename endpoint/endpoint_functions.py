from __future__ import annotations
from typing import Any, Literal
from PIL import Image
from pydantic import BaseModel, Field

class ClickAbsoluteAction(BaseModel):
    """
    Data model representing an absolute click action on a GUI.

    Attributes:
        action (Literal["click_absolute"]): The type of action, always "click_absolute".
        x (int): The x coordinate (pixels from the left edge).
        y (int): The y coordinate (pixels from the top edge).
    """
    action: Literal["click_absolute"] = "click_absolute"
    x: int = Field(description="The x coordinate, number of pixels from the left edge.")
    y: int = Field(description="The y coordinate, number of pixels from the top edge.")


def get_chat_messages(task: str, image: Image.Image) -> list[dict[str, Any]]:
    """
    Constructs a prompt for a navigation task, instructing the model to localize an element
    on a GUI image and output a click position in JSON format.

    Args:
        task (str): The description of the navigation task or target element.
        image (Image.Image): The GUI image to analyze.

    Returns:
        list[dict[str, Any]]: A list containing a single chat message dictionary structured for the model.
    """
    # Create the prompt string with instructions and JSON schema format
    prompt = (
        "Localize an element on the GUI image according to the provided target and output a click position.\n"
        f" * You must output a valid JSON following the format: {ClickAbsoluteAction.model_json_schema()}\n"
        " Your target is:"
    )

    # Return the chat message structure expected by the model
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{prompt}\n{task}"},
            ],
        },
    ]