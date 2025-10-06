from __future__ import annotations
import base64


def encode_image_to_data_uri(image_path: str, mime_type: str = "png") -> str:
    """
    Encodes an image file into a Data URI format.

    Args:
        image_path (str): The file path of the image to encode.
        mime_type (str): The MIME type of the image (default is "png").

    Returns:
        str: The image encoded as a Data URI string.
    """
    with open(image_path, "rb") as f:
        # Read the image file as binary and encode it in base64
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # Return the Data URI string
    return f"data:image/{mime_type};base64,{b64}"


def encode_image_to_base64(image_path): # for open ai api
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')