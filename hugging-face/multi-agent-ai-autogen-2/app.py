import gradio as gr
import datetime, os, threading

from multi_agent import run_multi_agent

lock = threading.Lock()

LLM = "gpt-4o"

def invoke(openai_api_key, task):
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")

    if not task:
        raise gr.Error("Task is required.")

    raise gr.Error("Please clone space due to local code execution.")
    
    with lock:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        result = run_multi_agent(LLM, task)
        del os.environ["OPENAI_API_KEY"]
        return result

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1),
                              gr.Textbox(label = "Task", value = f"Today is {datetime.date.today()}. {os.environ['INPUT']}")],
                    outputs = [gr.Markdown(label = "Output", value = os.environ["OUTPUT"], line_breaks = True, sanitize_html = False)],
                    title = "Multi-Agent AI: Coding",
                    description = os.environ["DESCRIPTION"])

demo.launch()
