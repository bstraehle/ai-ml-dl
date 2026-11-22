import logging
import os
import sys

import gradio as gr

from crew import get_crew

LLM = "gpt-6.2-sol"
VERBOSE = False
CONCURRENCY_LIMIT = 4

# Gradio's Markdown blocks scroll (overflow: auto) with no padding, which clips
# focus outlines on links at their edges
CSS = """
.block > div > .prose { padding: 6px; }
.prose a:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; border-radius: 4px; }
"""

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def invoke(openai_api_key: str, topic: str) -> str:
    """Research a topic on the web and write an edited, referenced article about it.

    Args:
        openai_api_key: OpenAI API key.
        topic: Topic to research and write about.
    """
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")
    if not topic:
        raise gr.Error("Topic is required.")

    try:
        article = str(get_crew(LLM, openai_api_key, VERBOSE).kickoff(inputs={"topic": topic}))
    except Exception as e:
        logger.exception("Request failed")
        raise gr.Error(str(e)) from e

    logger.info("Article:\n%s", article)
    return article


with gr.Blocks() as assistant:
    gr.Markdown("## Multi-Agent AI: Deep Research")
    gr.Markdown(os.environ.get("DESCRIPTION", ""))

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                openai_api_key = gr.Textbox(label="OpenAI API Key", type="password", lines=1)
                topic = gr.Textbox(label="Topic", value=os.environ.get("TOPIC", ""), lines=1)
            with gr.Row():
                clear_btn = gr.ClearButton()
                submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=3):
            article = gr.Markdown(label="Article", value=os.environ.get("OUTPUT", ""), line_breaks=True, sanitize_html=False)

    clear_btn.add([openai_api_key, topic, article])

    submit_btn.click(
        fn=invoke,
        inputs=[openai_api_key, topic],
        outputs=article,
        concurrency_limit=CONCURRENCY_LIMIT,
    )

if __name__ == "__main__":
    # Gradio 6's bundled SourceSansPro-SemiBold.woff2 is a 55-glyph subset, so
    # semibold text mixes fonts; load the full family from Google Fonts instead.
    theme = gr.themes.Default(
        font=[gr.themes.GoogleFont("Source Sans 3", weights=(400, 600, 700)), "ui-sans-serif", "system-ui", "sans-serif"]
    )
    assistant.launch(mcp_server=True, theme=theme, css=CSS)
