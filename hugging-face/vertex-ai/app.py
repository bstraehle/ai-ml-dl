import gradio as gr
import json, os, vertexai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

credentials = os.environ["CREDENTIALS"]
project = os.environ["PROJECT"]

config = {
    "max_output_tokens": 1000,
    "model": "gemini-1.5-pro-preview-0514",
    "temperature": 0.1,
    "top_k": 40,
    "top_p": 1.0,
}

credentials = json.loads(credentials)

from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_info(credentials)

if credentials.expired:
    credentials.refresh(Request())

vertexai.init(location = "us-central1",
              credentials = credentials,
              project = project
             )

from vertexai.preview.generative_models import GenerativeModel
generation_model = GenerativeModel(config["model"])

def invoke(prompt):
    if not prompt:
        raise gr.Error("Prompt is required.")

    raise gr.Error("Please clone and bring your own credentials.")
    
    completion = ""
    
    try:
        completion = generation_model.generate_content(prompt,
                                                       generation_config = {
                                                           "max_output_tokens": config["max_output_tokens"],
                                                           "temperature": config["temperature"],
                                                           "top_k": config["top_k"],
                                                           "top_p": config["top_p"]})
        
        if (completion.text != None):
            completion = completion.text
    except Exception as e:
        completion = e
        
        raise gr.Error(e)
    
    return completion

description = """<a href='https://www.gradio.app/'>Gradio</a> UI using the <a href='https://cloud.google.com/vertex-ai'>Google Vertex AI</a> SDK 
                 with Gemini 1.5 Pro model."""

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "Prompt", value = "If I dry one shirt in the sun, it takes 1 hour. How long do 3 shirts take?", lines = 1)],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    description = description)

demo.launch()
