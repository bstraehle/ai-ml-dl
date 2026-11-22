import base64
import logging
import mimetypes
import os
import sys

import gradio as gr
from openai import OpenAI

MODEL = "gpt-6.2-sol"
REASONING_EFFORT = "medium"
CONCURRENCY_LIMIT = 8

DEFAULT_PROMPT = "Describe the diagram"
DEFAULT_IMAGE = "https://raw.githubusercontent.com/bstraehle/ai-ml-dl/main/assets/architecture.png"

# Gradio's Markdown blocks (description, completion) scroll (overflow: auto) with no
# padding, which clips focus outlines at their edges
CSS = """
.block > div > .prose { padding: 6px; }
.prose a:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; border-radius: 4px; }
"""

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def image_data_url(path: str) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime_type};base64,{base64.b64encode(f.read()).decode()}"


def invoke(openai_api_key: str, prompt: str, image: str) -> str:
    """Answer a prompt about an image with OpenAI's GPT-6.2 Sol model.

    Args:
        openai_api_key: OpenAI API key.
        prompt: Question or instruction about the image.
        image: The image to reason about.

    Returns:
        The model's answer in Markdown.
    """
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")
    if not prompt:
        raise gr.Error("Prompt is required.")
    if not image:
        raise gr.Error("Image is required.")

    try:
        completion = OpenAI(api_key=openai_api_key).chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(image)}},
                ],
            }],
            reasoning_effort=REASONING_EFFORT,
        )
    except Exception as e:
        logger.exception("Request failed")
        raise gr.Error(str(e)) from e

    content = completion.choices[0].message.content

    if not content:
        raise gr.Error(f"The model returned no text (finish reason: {completion.choices[0].finish_reason}).")

    logger.info("Completion:\n%s", content)
    return content


demo = gr.Interface(
    fn=invoke,
    inputs=[
        gr.Textbox(label="OpenAI API Key", type="password", lines=1),
        gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=1),
        gr.Image(label="Image", type="filepath", sources=["upload"], value=DEFAULT_IMAGE, height=400),
    ],
    outputs=[gr.Markdown(label="Completion", value=os.environ.get("OUTPUT", ""))],
    title="Multimodal Reasoning Application",
    description=os.environ.get("DESCRIPTION", ""),
    concurrency_limit=CONCURRENCY_LIMIT,
)

if __name__ == "__main__":
    # Gradio 6's bundled SourceSansPro-SemiBold.woff2 is a 55-glyph subset, so
    # semibold text mixes fonts; load the full family from Google Fonts instead.
    theme = gr.themes.Default(
        font=[gr.themes.GoogleFont("Source Sans 3", weights=(400, 600, 700)), "ui-sans-serif", "system-ui", "sans-serif"]
    )
    demo.launch(mcp_server=True, theme=theme, css=CSS)
