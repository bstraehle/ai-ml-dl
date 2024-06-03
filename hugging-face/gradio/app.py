import gradio as gr
import os, threading

from openai import OpenAI

openai_api_key_lock = threading.Lock()
prompt_lock = threading.Lock()

config = {
    "max_tokens": 1000,
    "model": "gpt-4o",
    "temperature": 0
}

def invoke(openai_api_key, prompt):
    with openai_api_key_lock:
        if not openai_api_key:
            raise gr.Error("OpenAI API Key is required.")
    
    with prompt_lock:
        if not prompt:
            raise gr.Error("Prompt is required.")

    with openai_api_key_lock:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    content = ""
    
    try:
        client = OpenAI()
    
        completion = client.chat.completions.create(
            max_tokens = config["max_tokens"],
            messages = [{"role": "user", "content": prompt}],
            model = config["model"],
            temperature = config["temperature"])
    
        content = completion.choices[0].message.content
    except Exception as e:
        err_msg = e

        raise gr.Error(e)

    return content

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1),
                              gr.Textbox(label = "Prompt", lines = 1)],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    title = "Generative AI - LLM",
                    description = os.environ["DESCRIPTION"])

demo.launch()
