import openai, os

from langchain.callbacks import get_openai_callback
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

RAG_CHROMA  = "Chroma"
RAG_MONGODB = "MongoDB"

PDF_URL       = "https://arxiv.org/pdf/2303.08774.pdf"
WEB_URL       = "https://openai.com/research/gpt-4"
YOUTUBE_URL_1 = "https://www.youtube.com/watch?v=--khbXchTeE"
YOUTUBE_URL_2 = "https://www.youtube.com/watch?v=hdhZwyf24mE"

YOUTUBE_DIR = "/data/yt"
CHROMA_DIR  = "/data/db"

MONGODB_ATLAS_CLUSTER_URI = os.environ["MONGODB_ATLAS_CLUSTER_URI"]
MONGODB_DB_NAME           = "langchain_db"
MONGODB_COLLECTION_NAME   = "gpt-4"
MONGODB_INDEX_NAME        = "default"

LLM_CHAIN_PROMPT = PromptTemplate(
    input_variables = ["question"], 
    template = os.environ["LLM_TEMPLATE"])
RAG_CHAIN_PROMPT = PromptTemplate(
    input_variables = ["context", "question"], 
    template = os.environ["RAG_TEMPLATE"])

def load_documents():
    docs = []
    
    # PDF
    loader = PyPDFLoader(PDF_URL)
    docs.extend(loader.load())
    
    # Web
    loader = WebBaseLoader(WEB_URL)
    docs.extend(loader.load())
    
    # YouTube
    loader = GenericLoader(
        YoutubeAudioLoader(
            [YOUTUBE_URL_1, YOUTUBE_URL_2], 
            YOUTUBE_DIR), 
        OpenAIWhisperParser())
    docs.extend(loader.load())
    
    return docs

def split_documents(config, docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_overlap = config["chunk_overlap"],
        chunk_size = config["chunk_size"])
    
    return text_splitter.split_documents(docs)
    
def store_chroma(chunks):
    Chroma.from_documents(
        documents = chunks, 
        embedding = OpenAIEmbeddings(disallowed_special = ()), 
        persist_directory = CHROMA_DIR)

def store_mongodb(chunks):
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    collection = client[MONGODB_DB_NAME][MONGODB_COLLECTION_NAME]

    MongoDBAtlasVectorSearch.from_documents(
        documents = chunks,
        embedding = OpenAIEmbeddings(disallowed_special = ()),
        collection = collection,
        index_name = MONGODB_INDEX_NAME)

def rag_ingestion(config):
    docs = load_documents()
    
    chunks = split_documents(config, docs)
    
    store_chroma(chunks)
    store_mongodb(chunks)

def retrieve_chroma():
    return Chroma(
        embedding_function = OpenAIEmbeddings(disallowed_special = ()),
        persist_directory = CHROMA_DIR)

def retrieve_mongodb():
    return MongoDBAtlasVectorSearch.from_connection_string(
        MONGODB_ATLAS_CLUSTER_URI,
        MONGODB_DB_NAME + "." + MONGODB_COLLECTION_NAME,
        OpenAIEmbeddings(disallowed_special = ()),
        index_name = MONGODB_INDEX_NAME)

def get_llm(config):
    return ChatOpenAI(
        model_name = config["model_name"], 
        temperature = config["temperature"])

def llm_chain(config, prompt):
    llm_chain = LLMChain(
        llm = get_llm(config), 
        prompt = LLM_CHAIN_PROMPT)
    
    with get_openai_callback() as cb:
        completion = llm_chain.generate([{"question": prompt}])
    
    return completion, llm_chain, cb

def rag_chain(config, rag_option, prompt):
    llm = get_llm(config)

    if (rag_option == RAG_CHROMA):
        db = retrieve_chroma()
    elif (rag_option == RAG_MONGODB):
        db = retrieve_mongodb()

    rag_chain = RetrievalQA.from_chain_type(
        llm, 
        chain_type_kwargs = {"prompt": RAG_CHAIN_PROMPT,
                             "verbose": True}, 
        retriever = db.as_retriever(search_kwargs = {"k": config["k"]}), 
        return_source_documents = True)
    
    with get_openai_callback() as cb:
        completion = rag_chain({"query": prompt})

    return completion, rag_chain, cb
