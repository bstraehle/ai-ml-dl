import logging
import os
import sys
from pathlib import Path

import gradio as gr

from multi_agent import run_multi_agent

LLM = "gpt-6.2-sol"
CONCURRENCY_LIMIT = 8
# Example output shown before the first run; too large for a Space variable because of the inline image
EXAMPLE_OUTPUT = Path(__file__).with_name("example_output.md")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


async def invoke(openai_api_key: str, task: str) -> str:
    """Write and run Python code for a task with two GPT-6.2 Sol agents built with Microsoft Agent Framework: a
    Code Writer that writes and runs the code in OpenAI's sandboxed code interpreter and a Code Reviewer that checks
    the result, code quality, and plots, requesting changes until it approves. Returns the final images with the
    conversation.

    Args:
        openai_api_key: OpenAI API key.
        task: Coding task, e.g. creating a plot of stock data.
    """
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")
    if not task:
        raise gr.Error("Task is required.")

    try:
        result = await run_multi_agent(openai_api_key, LLM, task)
    except Exception as e:
        logger.exception("Task failed")
        # Agent Framework wraps the OpenAI error in a ChatClientException; show the original message
        raise gr.Error(str(e.__cause__ or e)) from e

    if not result:
        raise gr.Error("The model returned no output.")
    return result


with gr.Blocks() as demo:
    gr.Markdown("## Multi-Agent AI: Coding")
    gr.Markdown(os.environ.get("DESCRIPTION", ""))

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                openai_api_key = gr.Textbox(label="OpenAI API Key", type="password", lines=1)
                task = gr.Textbox(label="Task", value=os.environ.get("INPUT", ""), lines=1)
            with gr.Row():
                clear_btn = gr.ClearButton(components=[openai_api_key, task])
                submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=3):
            output = gr.Markdown(label="Output", value=EXAMPLE_OUTPUT.read_text(encoding="utf-8") if EXAMPLE_OUTPUT.exists() else os.environ.get("OUTPUT", ""), line_breaks=True, sanitize_html=False)

    clear_btn.add(output)
    submit_btn.click(fn=invoke, inputs=[openai_api_key, task], outputs=output, concurrency_limit=CONCURRENCY_LIMIT)

if __name__ == "__main__":
    # Gradio 6's bundled SourceSansPro-SemiBold.woff2 is a 55-glyph subset, so
    # semibold text mixes fonts; load the full family from Google Fonts instead.
    theme = gr.themes.Default(
        font=[gr.themes.GoogleFont("Source Sans 3", weights=(400, 600, 700)), "ui-sans-serif", "system-ui", "sans-serif"]
    )
    demo.launch(mcp_server=True, theme=theme)
