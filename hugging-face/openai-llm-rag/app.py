import gradio as gr
import os, time

from dotenv import load_dotenv, find_dotenv

from rag import run_llm_chain, run_rag_chain, run_rag_batch
from trace import trace_wandb

_ = load_dotenv(find_dotenv())

RUN_RAG_BATCH = False # load, split, embed, and store documents

config = {
    "chunk_overlap": 150,       # split documents
    "chunk_size": 1500,         # split documents
    "k": 3,                     # retrieve documents
    "model_name": "gpt-4-0314", # llm
    "temperature": 0,           # llm
}

RAG_OFF     = "Off"
RAG_CHROMA  = "Chroma"
RAG_MONGODB = "MongoDB"

def invoke(openai_api_key, rag_option, prompt):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (rag_option is None):
        raise gr.Error("Retrieval Augmented Generation is required.")
    if (prompt == ""):
        raise gr.Error("Prompt is required.")

    if (RUN_RAG_BATCH):
        run_rag_batch(config)
    
    chain = None
    completion = ""
    result = ""
    cb = ""
    err_msg = ""
    
    try:
        start_time_ms = round(time.time() * 1000)

        if (rag_option == RAG_OFF):
            completion, chain, cb = run_llm_chain(config, openai_api_key, prompt)
            
            if (completion.generations[0] != None and completion.generations[0][0] != None):
                result = completion.generations[0][0].text
        else:
            completion, chain, cb = run_rag_chain(config, openai_api_key, rag_option, prompt)

            result = completion["result"]
    except Exception as e:
        err_msg = e

        raise gr.Error(e)
    finally:
        end_time_ms = round(time.time() * 1000)
        
        trace_wandb(config,
                    rag_option == RAG_OFF, 
                    prompt, 
                    completion, 
                    result, 
                    chain, 
                    cb, 
                    err_msg, 
                    start_time_ms, 
                    end_time_ms)
    return result

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1, value = "sk-"), 
                              gr.Textbox(label = "Prompt", value = "What are GPT-4's media capabilities in 5 emojis and 1 sentence?", lines = 1),
                              gr.Radio([RAG_OFF, RAG_CHROMA, RAG_MONGODB], label = "Retrieval-Augmented Generation", value = RAG_OFF),],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    title = "Context-Aware Multimodal Reasoning Application",
                    description = os.environ["DESCRIPTION"],
                    examples = [["sk-", "What are GPT-4's media capabilities in 5 emojis and 1 sentence?", RAG_MONGODB],
                                ["sk-", "List GPT-4's exam scores and benchmark results.", RAG_CHROMA],
                                ["sk-", "Compare GPT-4 to GPT-3.5 in markdown table format.", RAG_MONGODB],
                                ["sk-", "Write a Python program that calls the GPT-4 API.", RAG_CHROMA],
                                ["sk-", "What is the GPT-4 API's cost and rate limit? Answer in English, Arabic, Chinese, Hindi, and Russian in JSON format.", RAG_MONGODB],],
                                cache_examples = False)

demo.launch()
