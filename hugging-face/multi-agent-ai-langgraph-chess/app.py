import logging
import os
import sys

import gradio as gr

from multi_agent import play

LLM_WHITE = "gpt-6.2-sol"
LLM_BLACK = "gpt-6-luna"
MAX_MOVES = 50
CONCURRENCY_LIMIT = 8

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


async def invoke(openai_api_key: str, num_moves: float = 10):
    """Play a chess game between two LLM agents (white: GPT-6.2 Sol, black: GPT-6 Luna) built with
    LangGraph, returning each move's analysis with an SVG board.

    Args:
        openai_api_key: OpenAI API key.
        num_moves: Number of half-moves to play (1-50).
    """
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")
    if not num_moves or not 1 <= num_moves <= MAX_MOVES:
        raise gr.Error(f"Number of Moves must be between 1 and {MAX_MOVES}.")

    result = ""
    try:
        async for move in play(openai_api_key, LLM_WHITE, LLM_BLACK, int(num_moves)):
            result += move
            yield result
    except Exception as e:
        logger.exception("Game failed")
        raise gr.Error(str(e)) from e

    logger.info("Game finished after %s requested moves", int(num_moves))


with gr.Blocks() as demo:
    gr.Markdown("## Multi-Agent AI: Chess")
    gr.Markdown(os.environ.get("DESCRIPTION", ""))

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                openai_api_key = gr.Textbox(label="OpenAI API Key", type="password", lines=1)
                num_moves = gr.Number(label="Number of Moves", value=10, minimum=1, maximum=MAX_MOVES, precision=0)
            with gr.Row():
                clear_btn = gr.ClearButton(components=[openai_api_key, num_moves])
                submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=2):
            game = gr.Markdown(label="Game", value=os.environ.get("OUTPUT", ""), line_breaks=True, sanitize_html=False)

    clear_btn.add(game)
    submit_btn.click(fn=invoke, inputs=[openai_api_key, num_moves], outputs=game, concurrency_limit=CONCURRENCY_LIMIT)

if __name__ == "__main__":
    # Gradio 6's bundled SourceSansPro-SemiBold.woff2 is a 55-glyph subset, so
    # semibold text mixes fonts; load the full family from Google Fonts instead.
    theme = gr.themes.Default(
        font=[gr.themes.GoogleFont("Source Sans 3", weights=(400, 600, 700)), "ui-sans-serif", "system-ui", "sans-serif"]
    )
    demo.launch(mcp_server=True, theme=theme)
