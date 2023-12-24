import openai, os

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

LLM_CHAIN_PROMPT = PromptTemplate(input_variables = ["question"], template = os.environ["LLM_TEMPLATE"])
RAG_CHAIN_PROMPT = PromptTemplate(input_variables = ["context", "question"], template = os.environ["RAG_TEMPLATE"])

client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
collection = client[MONGODB_DB_NAME][MONGODB_COLLECTION_NAME]

def document_loading():
    docs = []
    
    # PDF
    loader = PyPDFLoader(PDF_URL)
    docs.extend(loader.load())
    
    # Web
    loader = WebBaseLoader(WEB_URL)
    docs.extend(loader.load())
    
    # YouTube
    loader = GenericLoader(YoutubeAudioLoader([YOUTUBE_URL_1, YOUTUBE_URL_2], YOUTUBE_DIR), 
                           OpenAIWhisperParser())
    docs.extend(loader.load())
    
    return docs

def document_splitting(config, docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_overlap = config["chunk_overlap"],
                                                   chunk_size = config["chunk_size"])
    
    return text_splitter.split_documents(docs)
    
def document_storage_chroma(chunks):
    Chroma.from_documents(documents = chunks, 
                          embedding = OpenAIEmbeddings(disallowed_special = ()), 
                          persist_directory = CHROMA_DIR)

def document_storage_mongodb(chunks):
    MongoDBAtlasVectorSearch.from_documents(documents = chunks,
                                            embedding = OpenAIEmbeddings(disallowed_special = ()),
                                            collection = collection,
                                            index_name = MONGODB_INDEX_NAME)

def rag_batch(config):
    docs = document_loading()
    
    chunks = document_splitting(config, docs)
    
    document_storage_chroma(chunks)
    document_storage_mongodb(chunks)

def document_retrieval_chroma():
    return Chroma(embedding_function = OpenAIEmbeddings(disallowed_special = ()),
                  persist_directory = CHROMA_DIR)

def document_retrieval_mongodb():
    return MongoDBAtlasVectorSearch.from_connection_string(MONGODB_ATLAS_CLUSTER_URI,
                                                           MONGODB_DB_NAME + "." + MONGODB_COLLECTION_NAME,
                                                           OpenAIEmbeddings(disallowed_special = ()),
                                                           index_name = MONGODB_INDEX_NAME)

def get_llm(config, openai_api_key):
    return ChatOpenAI(model_name = config["model_name"], 
                      openai_api_key = openai_api_key, 
                      temperature = config["temperature"])

def llm_chain(config, openai_api_key, prompt):
    llm_chain = LLMChain(llm = get_llm(config, openai_api_key), 
                         prompt = LLM_CHAIN_PROMPT, 
                         verbose = False)
    
    completion = llm_chain.generate([{"question": prompt}])
    
    return completion, llm_chain

def rag_chain(config, openai_api_key, rag_option, prompt):
    llm = get_llm(config, openai_api_key)
    
    if (rag_option == RAG_CHROMA):
        db = document_retrieval_chroma()
    elif (rag_option == RAG_MONGODB):
        db = document_retrieval_mongodb()

    rag_chain = RetrievalQA.from_chain_type(llm, 
                                            chain_type_kwargs = {"prompt": RAG_CHAIN_PROMPT}, 
                                            retriever = db.as_retriever(search_kwargs = {"k": config["k"]}), 
                                            return_source_documents = True,
                                            verbose = False)
    
    completion = rag_chain({"query": prompt})
    
    return completion, rag_chain
