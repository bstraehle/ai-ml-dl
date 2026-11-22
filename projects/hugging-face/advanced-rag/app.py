import logging
import os
import re
import sys

import gradio as gr
from openai import OpenAI

from rag import inference, rag_inference, rag_retrieval_advanced, rag_retrieval_naive

RAG_OFF      = "Off"
RAG_NAIVE    = "Naive RAG"
RAG_ADVANCED = "Advanced RAG"

CONCURRENCY_LIMIT = 8

RAG_RETRIEVERS = {
    RAG_NAIVE: rag_retrieval_naive,
    RAG_ADVANCED: rag_retrieval_advanced,
}

# Markdown links (not already images) that point at Airbnb listing photos
PICTURE_LINK = re.compile(r"(?<!!)\[([^\]]*)\]\((https://a0\.muscache\.com/[^)\s]+)\)")

# Gradio's Markdown blocks (description, completion) scroll (overflow: auto) with no
# padding, which clips focus outlines at their edges
CSS = """
.block > div > .prose { padding: 6px; }
.prose a:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; border-radius: 4px; }
#completion img { max-width: 320px; max-height: 240px; object-fit: cover; border-radius: 8px; vertical-align: top; }
#completion a:has(img) { display: inline-block; border-radius: 8px; }
"""

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def inline_pictures(markdown: str) -> str:
    return PICTURE_LINK.sub(r"![\1](\2)", markdown)


def invoke(openai_api_key: str, prompt: str, accommodates: int, bedrooms: int, rag_option: str) -> str:
    """Recommend San Francisco Airbnb listings using retrieval-augmented generation.

    Args:
        openai_api_key: OpenAI API key.
        prompt: Question about Airbnb listings.
        accommodates: Number of guests the listing must accommodate.
        bedrooms: Number of bedrooms the listing must have.
        rag_option: One of "Off", "Naive RAG", or "Advanced RAG".
    """
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")
    if not prompt:
        raise gr.Error("Prompt is required.")
    if rag_option != RAG_OFF and rag_option not in RAG_RETRIEVERS:
        raise gr.Error("Retrieval-Augmented Generation is required.")

    client = OpenAI(api_key=openai_api_key)

    try:
        if rag_option == RAG_OFF:
            result = inference(client, prompt)
        else:
            retrieval_result = RAG_RETRIEVERS[rag_option](client, prompt, int(accommodates), int(bedrooms))
            result = rag_inference(client, prompt, retrieval_result, ranked=rag_option == RAG_ADVANCED)
    except Exception as e:
        logger.exception("Request failed")
        raise gr.Error(str(e)) from e

    logger.info("Completion:\n%s", result)
    return inline_pictures(result)


demo = gr.Interface(
    fn=invoke,
    inputs=[
        gr.Textbox(label="OpenAI API Key", type="password", lines=1),
        gr.Textbox(label="Prompt", value=os.environ.get("PROMPT", ""), lines=1),
        gr.Number(label="Accommodates", value=2, precision=0, minimum=1),
        gr.Number(label="Bedrooms", value=1, precision=0, minimum=0),
        gr.Radio([RAG_OFF, RAG_NAIVE, RAG_ADVANCED], label="Retrieval-Augmented Generation", value=RAG_ADVANCED),
    ],
    outputs=[gr.Markdown(label="Completion", value=os.environ.get("COMPLETION", ""), line_breaks=True, sanitize_html=False, elem_id="completion")],
    title="Context-Aware Reasoning Application",
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
