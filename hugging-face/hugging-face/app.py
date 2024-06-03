import os, threading
import gradio as gr
from transformers import pipeline

lock = threading.Lock()

pipe = pipeline("image-to-text",
                model="Salesforce/blip-image-captioning-base")

def exec(input):
    with lock:
        out = pipe(input)
        return out[0]["generated_text"]

demo = gr.Interface(exec,
                    inputs=gr.Image(type="pil",
                                    value="https://raw.githubusercontent.com/bstraehle/ai-ml-dl/main/hugging-face/hugging-face/beach.jpg"),
                    outputs=[gr.Textbox(label = "output", value=os.environ["OUTPUT"])],
                    description=os.environ["DESCRIPTION"])

demo.launch()
