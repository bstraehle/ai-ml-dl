import gradio as gr
import openai, os, time, wandb

from langchain.chains import LLMChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores import MongoDBAtlasVectorSearch

from pymongo import MongoClient

from wandb.sdk.data_types.trace_tree import Trace

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

WANDB_API_KEY = os.environ["WANDB_API_KEY"]

MONGODB_URI = os.environ["MONGODB_ATLAS_CLUSTER_URI"]
client = MongoClient(MONGODB_URI)
MONGODB_DB_NAME = "langchain_db"
MONGODB_COLLECTION_NAME = "gpt-4"
MONGODB_COLLECTION = client[MONGODB_DB_NAME][MONGODB_COLLECTION_NAME]
MONGODB_INDEX_NAME = "default"

description = os.environ["DESCRIPTION"]

config = {
    "chunk_overlap": 150,
    "chunk_size": 1500,
    "k": 3,
    "model": "gpt-4",
    "temperature": 0,
}

template = """If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "Thanks for using the 🧠 app - Bernd" at the end of the answer. """

llm_template = "Answer the question at the end. " + template + "Question: {question} Helpful Answer: "
rag_template = "Use the following pieces of context to answer the question at the end. " + template + "{context}. Question: {question} Helpful Answer: "

LLM_CHAIN_PROMPT = PromptTemplate(input_variables = ["question"], template = llm_template)
RAG_CHAIN_PROMPT = PromptTemplate(input_variables = ["context", "question"], template = rag_template)

CHROMA_DIR  = "/data/chroma"
YOUTUBE_DIR = "/data/youtube"

PDF_URL       = "https://arxiv.org/pdf/2303.08774.pdf"
WEB_URL       = "https://openai.com/research/gpt-4"
YOUTUBE_URL_1 = "https://www.youtube.com/watch?v=--khbXchTeE"
YOUTUBE_URL_2 = "https://www.youtube.com/watch?v=hdhZwyf24mE"
YOUTUBE_URL_3 = "https://www.youtube.com/watch?v=vw-KWfKwvTQ"

def document_loading_splitting():
    # Document loading
    docs = []
    # Load PDF
    loader = PyPDFLoader(PDF_URL)
    docs.extend(loader.load())
    # Load Web
    loader = WebBaseLoader(WEB_URL)
    docs.extend(loader.load())
    # Load YouTube
    loader = GenericLoader(YoutubeAudioLoader([YOUTUBE_URL_1,
                                               YOUTUBE_URL_2,
                                               YOUTUBE_URL_3], YOUTUBE_DIR), 
                           OpenAIWhisperParser())
    docs.extend(loader.load())
    # Document splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_overlap = config["chunk_overlap"],
                                                   chunk_size = config["chunk_size"])
    splits = text_splitter.split_documents(docs)
    return splits

def document_storage_chroma(splits):
    Chroma.from_documents(documents = splits, 
                          embedding = OpenAIEmbeddings(disallowed_special = ()), 
                          persist_directory = CHROMA_DIR)

def document_storage_mongodb(splits):
    MongoDBAtlasVectorSearch.from_documents(documents = splits,
                                            embedding = OpenAIEmbeddings(disallowed_special = ()),
                                            collection = MONGODB_COLLECTION,
                                            index_name = MONGODB_INDEX_NAME)

def document_retrieval_chroma(llm, prompt):
    db = Chroma(embedding_function = OpenAIEmbeddings(),
                persist_directory = CHROMA_DIR)
    return db

def document_retrieval_mongodb(llm, prompt):
    db = MongoDBAtlasVectorSearch.from_connection_string(MONGODB_URI,
                                                         MONGODB_DB_NAME + "." + MONGODB_COLLECTION_NAME,
                                                         OpenAIEmbeddings(disallowed_special = ()),
                                                         index_name = MONGODB_INDEX_NAME)
    return db

def llm_chain(llm, prompt):
    llm_chain = LLMChain(llm = llm, prompt = LLM_CHAIN_PROMPT, verbose = False)
    completion = llm_chain.run({"question": prompt})
    return completion, llm_chain

def rag_chain(llm, prompt, db):
    rag_chain = RetrievalQA.from_chain_type(llm, 
                                            chain_type_kwargs = {"prompt": RAG_CHAIN_PROMPT}, 
                                            retriever = db.as_retriever(search_kwargs = {"k": config["k"]}), 
                                            return_source_documents = True,
                                            verbose = False)
    completion = rag_chain({"query": prompt})
    return completion, rag_chain

def wandb_trace(rag_option, prompt, completion, chain, status_msg, start_time_ms, end_time_ms):
    wandb.init(project = "openai-llm-rag")
    if (rag_option == "Off" or str(status_msg) != ""):
        result = completion
    else:
        result = completion["result"]
        doc_meta_source_0 = completion["source_documents"][0].metadata["source"]
        doc_meta_source_1 = completion["source_documents"][1].metadata["source"]
        doc_meta_source_2 = completion["source_documents"][2].metadata["source"]
    trace = Trace(
        kind = "chain",
        name = type(chain).__name__ if (chain != None) else "",
        status_code = "SUCCESS" if (str(status_msg) == "") else "ERROR",
        status_message = str(status_msg),
        metadata={
            "chunk_overlap": "" if (rag_option == "Off") else config["chunk_overlap"],
            "chunk_size": "" if (rag_option == "Off") else config["chunk_size"],
            "k": "" if (rag_option == "Off") else config["k"],
            "model": config["model"],
            "temperature": config["temperature"],
        },
        inputs = {"rag_option": rag_option if (str(status_msg) == "") else "",
                  "prompt": str(prompt if (str(status_msg) == "") else ""), 
                  "prompt_template": str((llm_template if (rag_option == "Off") else rag_template) if (str(status_msg) == "") else ""),
                  "doc_meta_source_0": "" if (rag_option == "Off" or str(status_msg) != "") else str(doc_meta_source_0),
                  "doc_meta_source_1": "" if (rag_option == "Off" or str(status_msg) != "") else str(doc_meta_source_1),
                  "doc_meta_source_2": "" if (rag_option == "Off" or str(status_msg) != "") else str(doc_meta_source_2)},
        outputs = {"result": result},
        start_time_ms = start_time_ms,
        end_time_ms = end_time_ms
    )
    trace.log("test")
    wandb.finish()

def invoke(openai_api_key, rag_option, prompt):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (rag_option is None):
        raise gr.Error("Retrieval Augmented Generation is required.")
    if (prompt == ""):
        raise gr.Error("Prompt is required.")
    completion = ""
    result = ""
    chain = None
    status_msg = ""
    try:
        start_time_ms = round(time.time() * 1000)
        llm = ChatOpenAI(model_name = config["model"], 
                         openai_api_key = openai_api_key, 
                         temperature = config["temperature"])
        if (rag_option == "Chroma"):
            #splits = document_loading_splitting()
            #document_storage_chroma(splits)
            db = document_retrieval_chroma(llm, prompt)
            completion, chain = rag_chain(llm, prompt, db)
            result = completion["result"]
        elif (rag_option == "MongoDB"):
            #splits = document_loading_splitting()
            #document_storage_mongodb(splits)
            db = document_retrieval_mongodb(llm, prompt)
            completion, chain = rag_chain(llm, prompt, db)
            result = completion["result"]
        else:
            result, chain = llm_chain(llm, prompt)
            completion = result
    except Exception as e:
        status_msg = e
        raise gr.Error(e)
    finally:
        end_time_ms = round(time.time() * 1000)
        wandb_trace(rag_option, prompt, completion, chain, status_msg, start_time_ms, end_time_ms)
    return result

gr.close_all()
demo = gr.Interface(fn=invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", value = "sk-", lines = 1), 
                              gr.Radio(["Off", "Chroma", "MongoDB"], label="Retrieval Augmented Generation", value = "Off"),
                              gr.Textbox(label = "Prompt", value = "What is GPT-4?", lines = 1)],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    title = "Generative AI - LLM & RAG",
                    description = description)
demo.launch()
