# Import necessary libraries
from __future__ import annotations
from typing import Any
import torch
import gradio as gr
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from endpoint_functions import get_chat_messages

# Model name for loading the pre-trained weights
model_name = "Hcompany/Holo1.5-3B"  # Options: "Hcompany/Holo1.5-7B", "Hcompany/Holo1.5-72B"

# Load the model and processor for image-text tasks
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)


def predict(image: Image.Image, task: str) -> Any:
    """
    Processes the input image and navigation task, generates a prompt, and returns the model's response.

    Args:
        image (Image.Image): The input GUI image.
        task (str): The navigation task or target element description.

    Returns:
        Any: The model's decoded output, typically a JSON with click coordinates.
    """
    # Resize image according to model's image processor configuration
    image_processor_config = processor.image_processor
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=image_processor_config.patch_size * image_processor_config.merge_size,
        min_pixels=image_processor_config.min_pixels,
        max_pixels=image_processor_config.max_pixels,
    )

    # Resize the image using high-quality resampling
    processed_image: Image.Image = image.resize(
        size=(resized_width, resized_height),
        resample=Image.Resampling.LANCZOS
    )

    # Construct the prompt messages for the model
    messages: list[dict[str, Any]] = get_chat_messages(task, processed_image)

    # Apply the chat template to format the prompt for the model
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process inputs for the model
    inputs = processor(
        text=[text_prompt],
        images=[processed_image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate the model's response
    generated_ids = model.generate(**inputs, max_new_tokens=256)

    # Trim the input IDs from the generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the output to obtain the result
    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return result


# Set up the Gradio interface for interactive prediction
iface = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Textbox()],
    outputs=gr.JSON()
)


# Launch the Gradio app with sharing enabled
iface.launch(share=True)