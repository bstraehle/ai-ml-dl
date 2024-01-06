import gradio as gr
import logging, os, sys, time

from dotenv import load_dotenv, find_dotenv

from rag_langchain import LangChainRAG
from rag_llamaindex import LlamaIndexRAG
from trace import trace_wandb

_ = load_dotenv(find_dotenv())

RAG_INGESTION = False # load, split, embed, and store documents

RAG_OFF        = "Off"
RAG_LANGCHAIN  = "LangChain"
RAG_LLAMAINDEX = "LlamaIndex"

config = {
    "chunk_overlap": 100,       # split documents
    "chunk_size": 2000,         # split documents
    "k": 3,                     # retrieve documents
    "model_name": "gpt-4-0314", # llm
    "temperature": 0            # llm
}

logging.basicConfig(stream = sys.stdout, level = logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream = sys.stdout))

def invoke(openai_api_key, prompt, rag_option):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (prompt == ""):
        raise gr.Error("Prompt is required.")
    if (rag_option is None):
        raise gr.Error("Retrieval Augmented Generation is required.")

    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    if (RAG_INGESTION):
        if (rag_option == RAG_LANGCHAIN):
            rag = LangChainRAG()
            rag.ingestion(config)
        elif (rag_option == RAG_LLAMAINDEX):
            rag = LlamaIndexRAG()
            rag.ingestion(config)
    
    completion = ""
    result = ""
    callback = ""
    err_msg = ""
    
    try:
        start_time_ms = round(time.time() * 1000)

        if (rag_option == RAG_LANGCHAIN):
            rag = LangChainRAG()
            completion, chain, callback = rag.rag_chain(config, prompt)

            result = completion["result"]
        elif (rag_option == RAG_LLAMAINDEX):
            rag = LlamaIndexRAG()
            result = rag.retrieval(config, prompt)
        else:
            rag = LangChainRAG()
            completion, chain, callback = rag.llm_chain(config, prompt)
            
            if (completion.generations[0] != None and 
                completion.generations[0][0] != None):
                result = completion.generations[0][0].text
    except Exception as e:
        err_msg = e

        raise gr.Error(e)
    finally:
        end_time_ms = round(time.time() * 1000)
        
        trace_wandb(
            config,
            rag_option, 
            prompt, 
            completion, 
            result, 
            callback, 
            err_msg, 
            start_time_ms, 
            end_time_ms
        )

    return result

gr.close_all()

demo = gr.Interface(
    fn = invoke, 
    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1), 
              gr.Textbox(label = "Prompt", value = "What are GPT-4's media capabilities in 5 emojis and 1 sentence?", lines = 1),
              gr.Radio([RAG_OFF, RAG_LANGCHAIN, RAG_LLAMAINDEX], label = "Retrieval-Augmented Generation", value = RAG_LANGCHAIN)],
    outputs = [gr.Textbox(label = "Completion", lines = 1)],
    title = "Context-Aware Reasoning Application",
    description = os.environ["DESCRIPTION"],
    examples = [["", "What are GPT-4's media capabilities in 5 emojis and 1 sentence?", RAG_LANGCHAIN],
                ["", "List GPT-4's exam scores and benchmark results.", RAG_LLAMAINDEX],
                ["", "Compare GPT-4 to GPT-3.5 in markdown table format.", RAG_LANGCHAIN],
                ["", "Write a Python program that calls the GPT-4 API.", RAG_LLAMAINDEX],
                ["", "What is the GPT-4 API's cost and rate limit? Answer in English, Arabic, Chinese, Hindi, and Russian in JSON format.", RAG_LANGCHAIN]],
               cache_examples = False
)

demo.launch()
