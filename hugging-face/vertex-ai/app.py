import json
import logging
import os
import sys
from functools import cache

import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.oauth2 import service_account

load_dotenv()

MODEL = "gemini-4-argon"
LOCATION = "global"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
CONCURRENCY_LIMIT = 8
REQUIRE_OWN_CREDENTIALS = True

DEFAULT_PROMPT = "If I dry one shirt in the sun, it takes 1 hour. How long do 3 shirts take?"
DESCRIPTION = (
    "<a href='https://www.gradio.app/'>Gradio</a> UI using the "
    "<a href='https://cloud.google.com/vertex-ai'>Google Vertex AI</a> "
    "<a href='https://github.com/googleapis/python-genai'>Gen AI SDK</a> with Gemini 4 Argon model."
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
def get_client() -> genai.Client:
    credentials = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["CREDENTIALS"]), scopes=SCOPES
    )
    return genai.Client(enterprise=True, credentials=credentials, project=os.environ["PROJECT"], location=LOCATION)


def invoke(prompt: str) -> str:
    """Answer a prompt with Google's Gemini 4 Argon model on Vertex AI.

    Args:
        prompt: The prompt to send to Gemini.

    Returns:
        Gemini's answer in Markdown.
    """
    if not prompt:
        raise gr.Error("Prompt is required.")
    if REQUIRE_OWN_CREDENTIALS:
        raise gr.Error("Please clone and bring your own credentials.")

    try:
        response = get_client().models.generate_content(model=MODEL, contents=prompt)
    except Exception as e:
        logger.exception("Request failed")
        raise gr.Error(str(e)) from e

    if not response.text:
        finish_reason = response.candidates[0].finish_reason if response.candidates else None
        reason = finish_reason.value if finish_reason else "unknown"
        raise gr.Error(f"Gemini returned no text (finish reason: {reason}).")

    logger.info("Completion:\n%s", response.text)
    return response.text


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
