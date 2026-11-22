# References:

# https://www.stockfish.online/docs.php (API)
# https://www.stockfish.online/ (FEN to image)

import logging
import re
import sys

import gradio as gr
import httpx

STOCKFISH_API_URL = "https://stockfish.online/api/s/v2.php"
DEPTH = 12
TIMEOUT = 10
CONCURRENCY_LIMIT = 8

FEN_PATTERN = re.compile(r"[rnbqkpRNBQKP1-8/]+\s+[wb]\s+(?:-|[KQkq]+)\s+(?:-|[a-h][36])\s+\d+\s+\d+")
DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


# MCP server functions

def position_evaluation(fen: str) -> dict:
    """Chess position evaluation. Get best move with continuation in UCI notation for chess position in FEN.

    Args:
        fen: Chess position in FEN

    Returns:
        dict: Stockfish evaluation with "bestmove", "continuation" (best move with continuation in UCI notation),
        "evaluation", and "mate"
    """
    fen = (fen or "").strip()

    if not FEN_PATTERN.fullmatch(fen):
        raise gr.Error("Invalid FEN.")

    logger.info("position_evaluation: fen=%s", fen)

    try:
        response = httpx.get(STOCKFISH_API_URL, params={"fen": fen, "depth": DEPTH}, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPError as e:
        raise gr.Error(f"Stockfish API request failed: {e}") from e

    if not result.get("success"):
        raise gr.Error(result.get("error") or "Stockfish API returned no evaluation.")

    logger.info("position_evaluation: result=%s", result)

    return result


# Graphical user interface

mcp = gr.Interface(
    fn=position_evaluation,
    inputs=[gr.Textbox(label="Forsyth-Edwards Notation (FEN)", value=DEFAULT_FEN, max_length=100)],
    outputs=[gr.JSON(label="Evaluation")],
    title="Stockfish Chess Engine",
    description="Chess position evaluation by top open source chess engine",
    concurrency_limit=CONCURRENCY_LIMIT,
)

if __name__ == "__main__":
    # Gradio 6's bundled SourceSansPro-SemiBold.woff2 is a 55-glyph subset, so
    # semibold text mixes fonts; load the full family from Google Fonts instead.
    theme = gr.themes.Default(
        font=[gr.themes.GoogleFont("Source Sans 3", weights=(400, 600, 700)), "ui-sans-serif", "system-ui", "sans-serif"]
    )
    mcp.launch(mcp_server=True, ssr_mode=False, theme=theme)
