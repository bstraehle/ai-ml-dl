import gradio as gr
import json, os, vertexai, wandb

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

credentials = os.environ["CREDENTIALS"]
project = os.environ["PROJECT"]
wandb_api_key = os.environ["WANDB_API_KEY"]

config = {
    "model": "text-bison@001",
}

wandb.login(key = wandb_api_key)
wandb.init(project = "vertex-ai-llm", config = config)
config = wandb.config

credentials = json.loads(credentials)

from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_info(credentials)

if credentials.expired:
    credentials.refresh(Request())

vertexai.init(project = project, 
              location = "us-central1",
              credentials = credentials)

from vertexai.language_models import TextGenerationModel
generation_model = TextGenerationModel.from_pretrained(config.model)

def invoke(prompt):
    completion = generation_model.predict(prompt = prompt).text
    wandb.log({"prompt": prompt, "completion": completion})
    return completion

description = """<a href='https://www.gradio.app/'>Gradio</a> UI using <a href='https://cloud.google.com/vertex-ai?hl=en/'>Google Vertex AI</a> API 
                 with Bison foundation model. Model performance evaluation via <a href='https://wandb.ai/bstraehle'>Weights & Biases</a>."""

gr.close_all()
demo = gr.Interface(fn=invoke, 
                    inputs = [gr.Textbox(label = "Prompt", lines = 1)],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    title = "Generative AI - LLM",
                    description = description)
demo.launch()
