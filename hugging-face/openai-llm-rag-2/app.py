import gradio as gr
import os

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from rag import rag_ingestion, rag_retrieval

_ = load_dotenv(find_dotenv())

RAG_INGESTION = True # load, split, embed, and store documents

config = {
    "chunk_overlap": 150,       # split documents
    "chunk_size": 1500,         # split documents
    "k": 3,                     # retrieve documents
    "model_name": "gpt-4-0314", # llm
    "temperature": 0            # llm
}

RAG_OFF     = "Off"
RAG_MONGODB = "MongoDB"

def invoke(openai_api_key, prompt, rag_option):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (prompt == ""):
        raise gr.Error("Prompt is required.")
    if (rag_option is None):
        raise gr.Error("Retrieval Augmented Generation is required.")

    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    if (RAG_INGESTION):
        rag_ingestion()
    
    content = ""
    err_msg = ""
    
    try:
        if (rag_option == RAG_OFF):
            client = OpenAI()
    
            completion = client.chat.completions.create(
                messages = [{"role": "user", "content": prompt}],
                model = config["model_name"],
                temperature = config["temperature"])
    
            content = completion.choices[0].message.content
        else:
            content = rag_retrieval(prompt)
    except Exception as e:
        err_msg = e

        raise gr.Error(e)

    return content

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1), 
                              gr.Textbox(label = "Prompt", value = "What are GPT-4's media capabilities in 5 emojis and 1 sentence?", lines = 1),
                              gr.Radio([RAG_OFF, RAG_MONGODB], label = "Retrieval-Augmented Generation", value = RAG_MONGODB)],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    title = "Context-Aware Reasoning Application",
                    description = os.environ["DESCRIPTION"],
                    examples = [["", "What are GPT-4's media capabilities in 5 emojis and 1 sentence?", RAG_MONGODB],
                                ["", "List GPT-4's exam scores and benchmark results.", RAG_MONGODB],
                                ["", "Compare GPT-4 to GPT-3.5 in markdown table format.", RAG_MONGODB],
                                ["", "Write a Python program that calls the GPT-4 API.", RAG_MONGODB],
                                ["", "What is the GPT-4 API's cost and rate limit? Answer in English, Arabic, Chinese, Hindi, and Russian in JSON format.", RAG_MONGODB]],
                                cache_examples = False)

demo.launch()
