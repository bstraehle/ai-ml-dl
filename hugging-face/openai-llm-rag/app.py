import gradio as gr
import os, time

from dotenv import load_dotenv, find_dotenv

from rag import llm_chain, rag_chain, rag_batch
from trace import wandb_trace

_ = load_dotenv(find_dotenv())

RAG_BATCH = False # document loading, splitting, storage

config = {
    "chunk_overlap": 150,       # document splitting
    "chunk_size": 1500,         # document splitting
    "k": 3,                     # document retrieval
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

    if (RAG_BATCH):
        rag_batch(config)
    
    chain = None
    completion = ""
    result = ""
    generation_info = ""
    llm_output = ""
    err_msg = ""
    
    try:
        start_time_ms = round(time.time() * 1000)

        if (rag_option == RAG_OFF):
            completion, chain = llm_chain(config, openai_api_key, prompt)
            
            if (completion.generations[0] != None and completion.generations[0][0] != None):
                result = completion.generations[0][0].text
                generation_info = completion.generations[0][0].generation_info

            llm_output = completion.llm_output
        else:
            completion, chain = rag_chain(config, openai_api_key, rag_option, prompt)
            result = completion["result"]
    except Exception as e:
        err_msg = e

        raise gr.Error(e)
    finally:
        end_time_ms = round(time.time() * 1000)
        
        wandb_trace(config,
                    rag_option == RAG_OFF, 
                    prompt, 
                    completion, 
                    result, 
                    generation_info, 
                    llm_output, 
                    chain, 
                    err_msg, 
                    start_time_ms, 
                    end_time_ms)
    return result

gr.close_all()

demo = gr.Interface(fn=invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1), 
                              gr.Radio([RAG_OFF, RAG_CHROMA, RAG_MONGODB], label = "Retrieval Augmented Generation", value = RAG_OFF),
                              gr.Textbox(label = "Prompt", value = "What are GPT-4's media capabilities in 5 emojis and 1 sentence?", lines = 1),
                             ],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    title = "Context-Aware Multimodal Reasoning Application",
                    description = os.environ["DESCRIPTION"])

demo.launch()
