import gradio as gr
import os, threading

from multi_agent import run_multi_agent

lock = threading.Lock()

LLM_WHITE = "gpt-4o"
LLM_BLACK = "gpt-4o"

def invoke(openai_api_key, num_moves = 10):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")

    with lock:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        result = run_multi_agent(LLM_WHITE, LLM_BLACK, num_moves)
        del os.environ["OPENAI_API_KEY"]
        return result

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1),
                              gr.Number(label = "Number of Moves", value = 10, minimum = 1, maximum = 25)],
                    outputs = [gr.Markdown(label = "Game", value=os.environ["OUTPUT"], line_breaks=True, sanitize_html=False)],
                    title = "Multi-Agent AI: Chess",
                    description = os.environ["DESCRIPTION"])

demo.launch()
