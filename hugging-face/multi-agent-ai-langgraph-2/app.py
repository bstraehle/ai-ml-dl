import gradio as gr
import os, threading

from multi_agent import run_multi_agent

lock = threading.Lock()

os.environ["LANGCHAIN_PROJECT"] = "langgraph-multi-agent-chess"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

LLM_BOARD = "gpt-4o"
LLM_WHITE = "gpt-4o"
LLM_BLACK = "gpt-4o"

def invoke(openai_api_key, max_moves = 10):
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")

    with lock:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        result = run_multi_agent(LLM_BOARD, LLM_WHITE, LLM_BLACK, max_moves)
        del os.environ["OPENAI_API_KEY"]
        return result

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1),
                              gr.Number(label = "Maximum Number of Moves", value = 10, minimum = 1, maximum = 50)],
                    outputs = [gr.Markdown(label = "Game", value=os.environ["OUTPUT"], line_breaks=True, sanitize_html=False)],
                    title = "Multi-Agent AI: Chess",
                    description = os.environ["DESCRIPTION"])

demo.launch()
