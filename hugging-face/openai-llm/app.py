import gradio as gr
import openai, os, wandb

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]
wandb_api_key = os.environ["WANDB_API_KEY"]

config = {
    "max_tokens": 500,
    "model": "gpt-4",
    "temperature": 0,
}

wandb.login(key = wandb_api_key)
wandb.init(project = "openai-llm", config = config)
config = wandb.config

def invoke(prompt):
    response = openai.ChatCompletion.create(
        model = config.model,
        messages = [{"role": "user", "content": prompt}],
        temperature = config.temperature,
        max_tokens = config.max_tokens,
    )
    completion = response.choices[0].message["content"]
    wandb.log({"prompt": prompt, "completion": completion})
    return completion

description = """<a href='https://www.gradio.app/'>Gradio</a> UI using <a href='https://platform.openai.com/'>OpenAI</a> API with GPT 4 foundation model. 
                 Model performance evaluation via <a href='https://wandb.ai/bstraehle'>Weights & Biases</a>."""

gr.close_all()
demo = gr.Interface(fn=invoke, 
                    inputs = [gr.Textbox(label = "Prompt", lines = 1)],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    title = "Generative AI - LLM",
                    description = description)
demo.launch()
