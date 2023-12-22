import gradio as gr
import openai, os, time, wandb

from dotenv import load_dotenv, find_dotenv
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

_ = load_dotenv(find_dotenv())

PDF_URL       = "https://arxiv.org/pdf/2303.08774.pdf"
WEB_URL       = "https://openai.com/research/gpt-4"
YOUTUBE_URL_1 = "https://www.youtube.com/watch?v=--khbXchTeE"
YOUTUBE_URL_2 = "https://www.youtube.com/watch?v=hdhZwyf24mE"
YOUTUBE_URL_3 = "https://www.youtube.com/watch?v=vw-KWfKwvTQ"

YOUTUBE_DIR = "/data/youtube"
CHROMA_DIR  = "/data/chroma"

MONGODB_ATLAS_CLUSTER_URI = os.environ["MONGODB_ATLAS_CLUSTER_URI"]
MONGODB_DB_NAME           = "langchain_db"
MONGODB_COLLECTION_NAME   = "gpt-4"
MONGODB_INDEX_NAME        = "default"

LLM_CHAIN_PROMPT = PromptTemplate(input_variables = ["question"], template = os.environ["LLM_TEMPLATE"])
RAG_CHAIN_PROMPT = PromptTemplate(input_variables = ["context", "question"], template = os.environ["RAG_TEMPLATE"])

WANDB_API_KEY = os.environ["WANDB_API_KEY"]

RAG_OFF     = "Off"
RAG_CHROMA  = "Chroma"
RAG_MONGODB = "MongoDB"

client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
collection = client[MONGODB_DB_NAME][MONGODB_COLLECTION_NAME]

config = {
    "chunk_overlap": 150,
    "chunk_size": 1500,
    "k": 3,
    "model_name": "gpt-4",
    "temperature": 0,
}

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
    split_documents = text_splitter.split_documents(docs)
    
    return split_documents

def document_storage_chroma(documents):
    Chroma.from_documents(documents = documents, 
                          embedding = OpenAIEmbeddings(disallowed_special = ()), 
                          persist_directory = CHROMA_DIR)

def document_storage_mongodb(documents):
    MongoDBAtlasVectorSearch.from_documents(documents = documents,
                                            embedding = OpenAIEmbeddings(disallowed_special = ()),
                                            collection = collection,
                                            index_name = MONGODB_INDEX_NAME)

def document_retrieval_chroma(llm, prompt):
    return Chroma(embedding_function = OpenAIEmbeddings(),
                  persist_directory = CHROMA_DIR)

def document_retrieval_mongodb(llm, prompt):
    return MongoDBAtlasVectorSearch.from_connection_string(MONGODB_ATLAS_CLUSTER_URI,
                                                           MONGODB_DB_NAME + "." + MONGODB_COLLECTION_NAME,
                                                           OpenAIEmbeddings(disallowed_special = ()),
                                                           index_name = MONGODB_INDEX_NAME)

def llm_chain(llm, prompt):
    llm_chain = LLMChain(llm = llm, 
                         prompt = LLM_CHAIN_PROMPT, 
                         verbose = False)
    completion = llm_chain.generate([{"question": prompt}])
    return completion, llm_chain

def rag_chain(llm, prompt, db):
    rag_chain = RetrievalQA.from_chain_type(llm, 
                                            chain_type_kwargs = {"prompt": RAG_CHAIN_PROMPT}, 
                                            retriever = db.as_retriever(search_kwargs = {"k": config["k"]}), 
                                            return_source_documents = True,
                                            verbose = False)
    completion = rag_chain({"query": prompt})
    return completion, rag_chain

def wandb_trace(rag_option, prompt, completion, result, generation_info, llm_output, chain, err_msg, start_time_ms, end_time_ms):
    wandb.init(project = "openai-llm-rag")
    
    trace = Trace(
        kind = "chain",
        name = "" if (chain == None) else type(chain).__name__,
        status_code = "success" if (str(err_msg) == "") else "error",
        status_message = str(err_msg),
        metadata = {"chunk_overlap": "" if (rag_option == RAG_OFF) else config["chunk_overlap"],
                    "chunk_size": "" if (rag_option == RAG_OFF) else config["chunk_size"],
                   } if (str(err_msg) == "") else {},
        inputs = {"rag_option": rag_option,
                  "prompt": prompt,
                  "chain_prompt": (str(chain.prompt) if (rag_option == RAG_OFF) else 
                                   str(chain.combine_documents_chain.llm_chain.prompt)),
                  "source_documents": "" if (rag_option == RAG_OFF) else str([doc.metadata["source"] for doc in completion["source_documents"]]),
                 } if (str(err_msg) == "") else {},
        outputs = {"result": result,
                   "generation_info": str(generation_info),
                   "llm_output": str(llm_output),
                   "completion": str(completion),
                  } if (str(err_msg) == "") else {},
        model_dict = {"client": (str(chain.llm.client) if (rag_option == RAG_OFF) else
                                 str(chain.combine_documents_chain.llm_chain.llm.client)),
                      "model_name": (str(chain.llm.model_name) if (rag_option == RAG_OFF) else
                                     str(chain.combine_documents_chain.llm_chain.llm.model_name)),
                      "temperature": (str(chain.llm.temperature) if (rag_option == RAG_OFF) else
                                      str(chain.combine_documents_chain.llm_chain.llm.temperature)),
                      "retriever": ("" if (rag_option == RAG_OFF) else str(chain.retriever)),
                     } if (str(err_msg) == "") else {},
        start_time_ms = start_time_ms,
        end_time_ms = end_time_ms
    )
    
    trace.log("evaluation")
    wandb.finish()

def invoke(openai_api_key, rag_option, prompt):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (rag_option is None):
        raise gr.Error("Retrieval Augmented Generation is required.")
    if (prompt == ""):
        raise gr.Error("Prompt is required.")
    
    chain = None
    completion = ""
    result = ""
    generation_info = ""
    llm_output = ""
    err_msg = ""
    
    try:
        start_time_ms = round(time.time() * 1000)

        llm = ChatOpenAI(model_name = config["model_name"], 
                         openai_api_key = openai_api_key, 
                         temperature = config["temperature"])
        
        if (rag_option == RAG_CHROMA):
            #splits = document_loading_splitting()
            #document_storage_chroma(splits)
            
            db = document_retrieval_chroma(llm, prompt)
            completion, chain = rag_chain(llm, prompt, db)
            result = completion["result"]
        elif (rag_option == RAG_MONGODB):
            #splits = document_loading_splitting()
            #document_storage_mongodb(splits)
            
            db = document_retrieval_mongodb(llm, prompt)
            completion, chain = rag_chain(llm, prompt, db)
            result = completion["result"]
        else:
            completion, chain = llm_chain(llm, prompt)
            
            if (completion.generations[0] != None and completion.generations[0][0] != None):
                result = completion.generations[0][0].text
                generation_info = completion.generations[0][0].generation_info

            llm_output = completion.llm_output
    except Exception as e:
        err_msg = e
        raise gr.Error(e)
    finally:
        end_time_ms = round(time.time() * 1000)
        
        wandb_trace(rag_option, prompt, completion, result, generation_info, llm_output, chain, err_msg, start_time_ms, end_time_ms)
    return result

gr.close_all()
demo = gr.Interface(fn=invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1), 
                              gr.Radio([RAG_OFF, RAG_CHROMA, RAG_MONGODB], label = "Retrieval Augmented Generation", value = RAG_OFF),
                              gr.Textbox(label = "Prompt", value = "What are GPT-4's media capabilities in 5 emojis and 1 sentence?", lines = 1),
                             ],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    title = "Generative AI - LLM & RAG",
                    description = os.environ["DESCRIPTION"])
demo.launch()
