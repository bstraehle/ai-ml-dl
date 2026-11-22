import logging
import os
import sys

import gradio as gr

from multi_agent import run_multi_agent

LLM = "gpt-6.2-sol"
CONCURRENCY_LIMIT = 8

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


async def invoke(openai_api_key: str, topic: str) -> str:
    """Research a topic on the web and write an edited, referenced article about it, using Researcher,
    Author, and Editor agents (GPT-6.2 Sol) built with LangGraph.

    Args:
        openai_api_key: OpenAI API key.
        topic: Topic to research and write about.
    """
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")
    if not topic:
        raise gr.Error("Topic is required.")

    try:
        article = await run_multi_agent(openai_api_key, LLM, topic)
    except Exception as e:
        logger.exception("Research failed")
        raise gr.Error(str(e)) from e

    logger.info("Article on %r:\n=== BEGIN ARTICLE ===\n%s\n=== END ARTICLE ===", topic, article)
    return article


with gr.Blocks() as demo:
    gr.Markdown("## Multi-Agent AI: Deep Research")
    gr.Markdown(os.environ.get("DESCRIPTION", ""))

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                openai_api_key = gr.Textbox(label="OpenAI API Key", type="password", lines=1)
                topic = gr.Textbox(label="Topic", value=os.environ.get("TOPIC", ""), lines=1)
            with gr.Row():
                clear_btn = gr.ClearButton(components=[openai_api_key, topic])
                submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=3):
            article = gr.Markdown(label="Article", value=os.environ.get("OUTPUT", ""), line_breaks=True, sanitize_html=False)

    clear_btn.add(article)
    submit_btn.click(fn=invoke, inputs=[openai_api_key, topic], outputs=article, concurrency_limit=CONCURRENCY_LIMIT)

if __name__ == "__main__":
    # Gradio 6's bundled SourceSansPro-SemiBold.woff2 is a 55-glyph subset, so
    # semibold text mixes fonts; load the full family from Google Fonts instead.
    theme = gr.themes.Default(
        font=[gr.themes.GoogleFont("Source Sans 3", weights=(400, 600, 700)), "ui-sans-serif", "system-ui", "sans-serif"]
    )
    demo.launch(mcp_server=True, theme=theme)
