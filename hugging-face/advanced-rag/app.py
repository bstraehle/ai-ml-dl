#TODO: Pydantic, image embedding, clean up data set

import gradio as gr
import logging, os, sys, threading

from custom_utils import (
    connect_to_database,
    inference,
    rag_ingestion,
    rag_retrieval_naive,
    rag_retrieval_advanced,
    rag_inference
)

lock = threading.Lock()

RAG_INGESTION = False

RAG_OFF      = "Off"
RAG_NAIVE    = "Naive RAG"
RAG_ADVANCED = "Advanced RAG"

logging.basicConfig(stream = sys.stdout, level = logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream = sys.stdout))
    
def invoke(openai_api_key, 
           prompt, 
           accomodates, 
           bedrooms, 
           rag_option):
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")
    if not prompt:
        raise gr.Error("Prompt is required.")
    if not rag_option:
        raise gr.Error("Retrieval-Augmented Generation is required.")

    with lock:
        db, collection = connect_to_database()

        inference_result = ""
        
        if (RAG_INGESTION):
            return rag_ingestion(collection)
        elif rag_option == RAG_OFF:
            inference_result = inference(
                openai_api_key, 
                prompt
            )
        elif rag_option == RAG_NAIVE:
            retrieval_result = rag_retrieval_naive(
                openai_api_key, 
                prompt,
                db, 
                collection
            )
            inference_result = rag_inference(
                openai_api_key, 
                prompt, 
                retrieval_result
            )        
        elif rag_option == RAG_ADVANCED:
            retrieval_result = rag_retrieval_advanced(
                openai_api_key, 
                prompt, 
                accomodates,
                bedrooms,
                db, 
                collection
            )
            inference_result = rag_inference(
                openai_api_key, 
                prompt, 
                retrieval_result
            )

        print("###")
        print(inference_result)
        print("###")
 
        return inference_result

gr.close_all()

demo = gr.Interface(
    fn = invoke, 
    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1), 
              gr.Textbox(label = "Prompt", value = os.environ["PROMPT"], lines = 1),
              gr.Number(label = "Accomodates", value = 2),
              gr.Number(label = "Bedrooms", value = 1),
              gr.Radio([RAG_OFF, RAG_NAIVE, RAG_ADVANCED], label = "Retrieval-Augmented Generation", value = RAG_ADVANCED)],
    outputs = [gr.Markdown(label = "Completion", value = os.environ["COMPLETION"], line_breaks = True, sanitize_html = False)],
    title = "Context-Aware Reasoning Application",
    description = os.environ["DESCRIPTION"]
)

demo.launch()
