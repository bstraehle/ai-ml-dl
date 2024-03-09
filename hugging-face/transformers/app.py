import gradio as gr
from transformers import pipeline

pipe = pipeline("image-to-text",
                model="Salesforce/blip-image-captioning-base")

def exec(input):
    out = pipe(input)
    return out[0]["generated_text"]

desc = """<a href='https://www.gradio.app/'>Gradio</a> UI using the Hugging Face 
          <a href='https://huggingface.co/docs/transformers/en/index'>Transformers</a> 
          library for image captioning <a href='https://huggingface.co/tasks'>task</a>."""

demo = gr.Interface(exec,
                    inputs=gr.Image(type="pil",
                                    value="https://raw.githubusercontent.com/bstraehle/ai-ml-dl/main/hugging-face/transformers/beach.jpg"),
                    outputs="text",
                    description=desc)

demo.launch()
