import logging
import os
import sys
from functools import cache

import boto3
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

MODEL = "us.anthropic.claude-opus-5-5"
REGION = "us-east-1"
MAX_TOKENS = 16000
SYSTEM_PROMPT = "You are a honest, helpful, and harmless bot."
CONCURRENCY_LIMIT = 8
REQUIRE_OWN_CREDENTIALS = True

DEFAULT_PROMPT = "If I dry one shirt in the sun, it takes 1 hour. How long do 3 shirts take?"
DESCRIPTION = (
    "<a href='https://www.gradio.app/'>Gradio</a> UI using the "
    "<a href='https://aws.amazon.com/bedrock/'>Amazon Bedrock</a> SDK with "
    "<a href='https://www.anthropic.com/'>Anthropic</a> Claude Opus 5.5 model."
)

# Gradio's Markdown blocks (description, completion) scroll (overflow: auto) with no
# padding, which clips focus outlines at their edges
CSS = """
.block > div > .prose { padding: 6px; }
.prose a:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; border-radius: 4px; }
"""

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


@cache
def get_client():
    return boto3.client("bedrock-runtime", region_name=REGION)


def invoke(prompt: str) -> str:
    """Answer a prompt with Anthropic's Claude Opus 5.5 model on Amazon Bedrock.

    Args:
        prompt: The prompt to send to Claude.

    Returns:
        Claude's answer in Markdown.
    """
    if not prompt:
        raise gr.Error("Prompt is required.")
    if REQUIRE_OWN_CREDENTIALS:
        raise gr.Error("Please clone and bring your own credentials.")

    try:
        response = get_client().converse(
            modelId=MODEL,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": MAX_TOKENS},
        )
    except Exception as e:
        logger.exception("Request failed")
        raise gr.Error(str(e)) from e

    completion = "".join(block["text"] for block in response["output"]["message"]["content"] if "text" in block)

    if not completion:
        raise gr.Error(f"Claude returned no text (stop reason: {response.get('stopReason', 'unknown')}).")

    logger.info("Completion:\n%s", completion)
    return completion


demo = gr.Interface(
    fn=invoke,
    inputs=[gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=1)],
    outputs=[gr.Markdown(label="Completion", value=os.environ.get("COMPLETION", ""))],
    description=DESCRIPTION,
    concurrency_limit=CONCURRENCY_LIMIT,
)

if __name__ == "__main__":
    # Gradio 6's bundled SourceSansPro-SemiBold.woff2 is a 55-glyph subset, so
    # semibold text mixes fonts; load the full family from Google Fonts instead.
    theme = gr.themes.Default(
        font=[gr.themes.GoogleFont("Source Sans 3", weights=(400, 600, 700)), "ui-sans-serif", "system-ui", "sans-serif"]
    )
    demo.launch(mcp_server=True, theme=theme, css=CSS)
