import gradio as gr
import os, threading

from multi_agent import run_multi_agent

lock = threading.Lock()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-multi-agent-chess"

def invoke(openai_api_key, max_moves = 10):
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")

    with lock:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        result = run_multi_agent(max_moves)
        del os.environ["OPENAI_API_KEY"]
        return result

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1),
                              gr.Number(label = "Maximum Number of Moves (1-25)", value = 10, minimum = 1, maximum = 25)],
                    outputs = [gr.Markdown(label = "Game", value=os.environ["OUTPUT"], line_breaks=True, sanitize_html=False)],
                    title = "Multi-Agent AI: Chess",
                    description = os.environ["DESCRIPTION"])

demo.launch()
