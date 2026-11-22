import asyncio
import json
import logging
import os
import sys

import gradio as gr
from openai import AsyncOpenAI

from agent_langchain import agent_langchain
from agent_llamaindex import agent_llamaindex
from common import MODEL, REASONING_EFFORT, TIMEOUT

AGENT_OFF = "Off"
AGENT_LANGCHAIN = "LangChain"
AGENT_LLAMAINDEX = "LlamaIndex"

AGENTS = {
    AGENT_LANGCHAIN: agent_langchain,
    AGENT_LLAMAINDEX: agent_llamaindex,
}

AGENT_TIMEOUT = 120
CONCURRENCY_LIMIT = 8

DEFAULT_PROMPT = (
    "How does current weather in San Francisco and Paris compare in metric and imperial system? "
    "Answer in JSON format and include today's date."
)

# Gradio's Markdown blocks (description, completion) scroll (overflow: auto) with no
# padding, which clips focus outlines at their edges
CSS = """
.block > div > .prose { padding: 6px; }
.prose a:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; border-radius: 4px; }
"""

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def fence_json(text: str) -> str:
    """Wrap the first bare JSON object or array in a json code block so Markdown keeps its layout."""
    if "```" in text:
        return text
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(value, dict) or (value and all(isinstance(item, (dict, list)) for item in value)):
            return f"{text[:start].rstrip()}\n\n```json\n{text[start:end]}\n```\n\n{text[end:].lstrip()}".strip()
    return text


async def inference(openai_api_key: str, prompt: str) -> str:
    completion = await AsyncOpenAI(api_key=openai_api_key, timeout=TIMEOUT).chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=REASONING_EFFORT,
    )
    return completion.choices[0].message.content


async def invoke(openai_api_key: str, prompt: str, agent_option: str) -> str:
    """Answer a prompt with GPT-6 Luna, optionally as a LangChain or LlamaIndex agent with weather and date tools.

    Args:
        openai_api_key: OpenAI API key.
        prompt: Question to answer, e.g. about current weather.
        agent_option: One of "Off", "LangChain", or "LlamaIndex".

    Returns:
        The answer in Markdown.
    """
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")
    if not prompt:
        raise gr.Error("Prompt is required.")
    if agent_option != AGENT_OFF and agent_option not in AGENTS:
        raise gr.Error("Use Agent is required.")

    try:
        if agent_option == AGENT_OFF:
            result = await inference(openai_api_key, prompt)
        else:
            result = await asyncio.wait_for(AGENTS[agent_option](openai_api_key, prompt), AGENT_TIMEOUT)
    except TimeoutError as e:
        logger.exception("Agent timed out")
        raise gr.Error(f"The agent didn't finish within {AGENT_TIMEOUT} seconds.") from e
    except Exception as e:
        logger.exception("Request failed")
        raise gr.Error(str(e)) from e

    if not result:
        raise gr.Error("The model returned no text.")

    logger.info("Completion (%s):\n%s", agent_option, result)
    return fence_json(result)


demo = gr.Interface(
    fn=invoke,
    inputs=[
        gr.Textbox(label="OpenAI API Key", type="password", lines=1),
        gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=1),
        gr.Radio([AGENT_OFF, AGENT_LANGCHAIN, AGENT_LLAMAINDEX], label="Use Agent", value=AGENT_LANGCHAIN),
    ],
    outputs=[gr.Markdown(label="Completion", value=os.environ.get("OUTPUT", ""))],
    title="Agentic Reasoning Application",
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
