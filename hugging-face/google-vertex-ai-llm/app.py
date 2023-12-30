import gradio as gr
import json, os, vertexai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

credentials = os.environ["CREDENTIALS"]
project = os.environ["PROJECT"]

config = {
    "max_output_tokens": 1000,
    "model": "gemini-pro",
    "temperature": 0.1,
    "top_k": 40,
    "top_p": 1.0,
}

credentials = json.loads(credentials)

from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_info(credentials)

if credentials.expired:
    credentials.refresh(Request())

vertexai.init(project = project, 
              location = "us-central1",
              credentials = credentials
             )

from vertexai.preview.generative_models import GenerativeModel
generation_model = GenerativeModel(config["model"])

def invoke(prompt):
    if (prompt == ""):
        raise gr.Error("Prompt is required.")

    completion = ""
    
    try:
        completion = generation_model.generate_content(prompt,
                                                       generation_config = {
                                                           "max_output_tokens": config["max_output_tokens"],
                                                           "temperature": config["temperature"],
                                                           "top_k": config["top_k"],
                                                           "top_p": config["top_p"],
                                                       })
        
        if (completion.text != None):
            completion = completion.text
    except Exception as e:
        completion = e
        
        raise gr.Error(e)
    
    return completion

description = """<a href='https://www.gradio.app/'>Gradio</a> UI using the <a href='https://cloud.google.com/vertex-ai'>Google Vertex AI</a> API 
                 with Gemini Pro model."""

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "Prompt", lines = 1)],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    title = "Generative AI - LLM",
                    description = description)

demo.launch()
