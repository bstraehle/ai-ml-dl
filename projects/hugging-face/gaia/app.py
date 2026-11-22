# References:

# https://www.gradio.app/guides/quickstart

import os
import queue
import sys
import threading
import time

import gradio as gr

from agents.crew import run_crew
from utils.utils import (
    DATASET_TYPE_GAIA,
    DATASET_TYPE_HLE,
    MAX_API_KEY_LENGTH,
    get_dataset,
    validate_input
)

# Configuration

SPACE_ID = os.environ.get("SPACE_ID", "bstraehle/gaia")

BASE_URL = f"https://huggingface.co/spaces/{SPACE_ID}/blob/main"

# Console capture swaps the process-wide sys.stdout/sys.stderr, so runs must not overlap
CONCURRENCY_LIMIT = 1

# Streaming console output capture

class _Tee:
    def __init__(self, original, q):
        self._original = original
        self._queue = q

    def write(self, data):
        if data:
            self._queue.put(data)
        return self._original.write(data)

    def flush(self):
        self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


class ConsoleCapture:
    """Copy everything written to stdout/stderr into a queue until the block exits."""

    def __init__(self):
        self._queue = queue.Queue()

    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._stdout, self._queue)
        sys.stderr = _Tee(self._stderr, self._queue)
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def get_new_output(self):
        output = []

        while True:
            try:
                output.append(self._queue.get_nowait())
            except queue.Empty:
                return "".join(output)

# MCP server functions

def ask(oauth_token: gr.OAuthToken, question: str, openai_api_key: str, gemini_api_key: str, anthropic_api_key: str, file_name: str = ""):
    """
    Ask General AI Assistant a question to answer.
    Args:
        question (str): The question to answer
        openai_api_key (str): OpenAI API key (always used)
        gemini_api_key (str): Gemini API key (always used)
        anthropic_api_key (str): Anthropic API key (only used by Stagehand tool)
        file_name (str): Optional file name
    Yields:
        tuple: (answer, console_logs) - The answer to the question and captured console output (streamed)
    """
    openai_api_key, gemini_api_key, anthropic_api_key = (
        (k or "").strip() for k in (openai_api_key, gemini_api_key, anthropic_api_key)
    )

    required = {
        "Question": question,
        "OpenAI API Key": openai_api_key,
        "Gemini API Key": gemini_api_key,
        "Anthropic API Key": anthropic_api_key,
    }

    for label, value in required.items():
        if not value:
            msg = f"{label} is required."
            gr.Warning(msg)
            yield None, msg
            return

    if not validate_input(question, openai_api_key, gemini_api_key, anthropic_api_key):
        msg = "Invalid input"
        gr.Warning(msg)
        yield None, msg
        return

    task_file_path = f"files/{file_name}" if file_name else ""

    result = {"answer": None, "error": None}
    all_logs = ""

    def run_task():
        try:
            result["answer"] = run_crew(question, task_file_path, openai_api_key, gemini_api_key, anthropic_api_key)
        except Exception as e:
            result["error"] = str(e)

    try:
        with ConsoleCapture() as capture:
            thread = threading.Thread(target=run_task, daemon=True)
            thread.start()

            # Stream logs while waiting
            while thread.is_alive():
                new_output = capture.get_new_output()
                if new_output:
                    all_logs += new_output
                    yield None, all_logs
                time.sleep(0.1)

            all_logs += capture.get_new_output()
    except Exception as e:
        msg = str(e)
        gr.Warning(msg)
        yield None, f"{all_logs}\n\nError: {msg}"
        return

    if result["error"]:
        gr.Warning(result["error"])
        yield None, all_logs + f"\n\nError: {result['error']}"
    else:
        yield result["answer"], all_logs

# Helper functions

def update_file_link(file_name):
    if file_name:
        return f"<a href='{BASE_URL}/files/{file_name}' target='_blank'>Open File</a>"

    return ""

# Graphical user interface

DESCRIPTION = (
    f"Prototype <strong>multi-agent AI platform</strong> with high autonomy, "
    f"including code generation & execution, browser automation, and multi-modal reasoning. "
    f"The system can solve <a href='https://arxiv.org/pdf/2311.12983'>GAIA Benchmark</a> "
    f"Level 1, 2, and 3 problems as well as <a href='https://arxiv.org/pdf/2501.14249'>Humanity's Last Exam</a> "
    f"problems. To get started, select from the examples below. "
    f"Processing can take minutes depending on question complexity. "
    f"Console logs are provided below for transparency. "
    f"<a href='{BASE_URL}/README.md'>Architecture</a></p>"
)

DEFAULT_QUESTION = "How many GitHub repos does the person who submitted the 'General AI Assistant' solution in MCP's 1st Birthday Hackathon have?"

DEFAULT_GROUND_TRUTH = "20"

# Gradio's Markdown blocks scroll (overflow: auto) with no padding, which clips
# focus outlines on links at their edges
CSS = """
html,
body,
main,
.gradio-app {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow-x: hidden !important;
}
.full-width-app {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
.content-padding {
    padding: 0 1.5rem 0;
}
.block > div > .prose { padding: 6px; }
.prose a:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; border-radius: 4px; }
"""

EXAMPLE_TABS = [
    ("GAIA Benchmark Level 1", DATASET_TYPE_GAIA, 1),
    ("GAIA Benchmark Level 2", DATASET_TYPE_GAIA, 2),
    ("GAIA Benchmark Level 3", DATASET_TYPE_GAIA, 3),
    ("Humanity's Last Exam", DATASET_TYPE_HLE, 0),
]

with gr.Blocks(elem_classes=["full-width-app"]) as gaia:
    with gr.Column(elem_classes=["content-padding"]):
        gr.Markdown("## General AI Assistant")

        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    question = gr.Textbox(
                        label="Question *",
                        value=DEFAULT_QUESTION,
                        interactive=True,
                        max_length=500,
                        lines=1,
                        max_lines=5
                    )
                with gr.Row():
                    ground_truth = gr.Textbox(
                        label="Ground Truth",
                        value=DEFAULT_GROUND_TRUTH,
                        interactive=False,
                        lines=1,
                        max_lines=2
                    )
                    file_name = gr.Textbox(
                        label="File Name",
                        interactive=False,
                        lines=1,
                        max_lines=2,
                        scale=2
                    )
                    file_link = gr.HTML(
                        label="File Link",
                        value=""
                    )
                with gr.Row():
                    openai_api_key = gr.Textbox(
                        label="OpenAI API Key *",
                        type="password",
                        placeholder="sk‑...",
                        value="",
                        interactive=True,
                        max_length=MAX_API_KEY_LENGTH
                    )
                    gemini_api_key = gr.Textbox(
                        label="Gemini API Key *",
                        type="password",
                        value="",
                        interactive=True,
                        max_length=MAX_API_KEY_LENGTH
                    )
                    anthropic_api_key = gr.Textbox(
                        label="Anthropic API Key *",
                        type="password",
                        placeholder="sk-ant-...",
                        value="",
                        interactive=True,
                        max_length=MAX_API_KEY_LENGTH
                    )
                with gr.Row():
                    gr.LoginButton()
                    submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                answer = gr.Textbox(
                    label="Answer",
                    interactive=False,
                    lines=1,
                    max_lines=5
                )

        with gr.Accordion("Console Logs", open=False):
            console_logs = gr.Textbox(
                label="Output",
                interactive=False,
                lines=25,
                max_lines=25,
                autoscroll=True
            )

        with gr.Row():
            clear_btn = gr.ClearButton(
                components=[question, ground_truth, file_name, file_link, answer, console_logs]
            )

        submit_btn.click(
            fn=ask,
            inputs=[question, openai_api_key, gemini_api_key, anthropic_api_key, file_name],
            outputs=[answer, console_logs],
            concurrency_limit=CONCURRENCY_LIMIT
        )

        file_name.change(
            fn=update_file_link,
            inputs=[file_name],
            outputs=[file_link]
        )

        with gr.Tabs():
            for tab_label, dataset_type, level in EXAMPLE_TABS:
                with gr.TabItem(tab_label):
                    gr.Examples(
                        examples=get_dataset(dataset_type, level),
                        inputs=[question, ground_truth, file_name],
                        examples_per_page=5,
                        cache_examples=False
                    )

if __name__ == "__main__":
    # Gradio 6's bundled SourceSansPro-SemiBold.woff2 is a 55-glyph subset, so
    # semibold text mixes fonts; load the full family from Google Fonts instead.
    theme = gr.themes.Default(
        font=[gr.themes.GoogleFont("Source Sans 3", weights=(400, 600, 700)), "ui-sans-serif", "system-ui", "sans-serif"]
    )
    gaia.launch(mcp_server=True, ssr_mode=False, theme=theme, css=CSS)
