import os

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
from rag_base import BaseRAG

class LangChainRAG(BaseRAG):
    MONGODB_DB_NAME = "langchain_db"
    
    CHROMA_DIR  = "/data/db"
    YOUTUBE_DIR = "/data/yt"

    LLM_CHAIN_PROMPT = PromptTemplate(
        input_variables = ["question"], 
        template = os.environ["LLM_TEMPLATE"])
    RAG_CHAIN_PROMPT = PromptTemplate(
        input_variables = ["context", "question"], 
        template = os.environ["RAG_TEMPLATE"])

    def load_documents(self):
        docs = []
    
        # PDF
        loader = PyPDFLoader(self.PDF_URL)
        docs.extend(loader.load())
        #print("docs = " + str(len(docs)))
    
        # Web
        loader = WebBaseLoader(self.WEB_URL)
        docs.extend(loader.load())
        #print("docs = " + str(len(docs)))
    
        # YouTube
        loader = GenericLoader(
            YoutubeAudioLoader(
                [self.YOUTUBE_URL_1, self.YOUTUBE_URL_2], 
                self.YOUTUBE_DIR), 
            OpenAIWhisperParser())
        docs.extend(loader.load())
        #print("docs = " + str(len(docs)))
    
        return docs

    def split_documents(self, config, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_overlap  = config["chunk_overlap"],
            chunk_size = config["chunk_size"]
        )
    
        return text_splitter.split_documents(docs)
    
    def store_documents_chroma(self, chunks):
        Chroma.from_documents(
            documents = chunks, 
            embedding = OpenAIEmbeddings(disallowed_special = ()), # embed
            persist_directory = self.CHROMA_DIR
        )

    def store_documents_mongodb(self, chunks):
        client = MongoClient(self.MONGODB_ATLAS_CLUSTER_URI)
        collection = client[self.MONGODB_DB_NAME][self.MONGODB_COLLECTION_NAME]

        MongoDBAtlasVectorSearch.from_documents(
            documents = chunks,
            embedding = OpenAIEmbeddings(disallowed_special = ()),
            collection = collection,
            index_name = self.MONGODB_INDEX_NAME
        )

    def ingestion(self, config):
        docs = self.load_documents()
    
        chunks = self.split_documents(config, docs)
    
        #self.store_documents_chroma(chunks)
        self.store_documents_mongodb(chunks)

    def get_vector_store_chroma(self):
        return Chroma(
            embedding_function = OpenAIEmbeddings(disallowed_special = ()), # embed
            persist_directory = self.CHROMA_DIR
        )

    def get_vector_store_mongodb(self):
        return MongoDBAtlasVectorSearch.from_connection_string(
            self.MONGODB_ATLAS_CLUSTER_URI,
            self.MONGODB_DB_NAME + "." + self.MONGODB_COLLECTION_NAME,
            OpenAIEmbeddings(disallowed_special = ()),
            index_name = self.MONGODB_INDEX_NAME
        )

    def get_llm(self, config):
        return ChatOpenAI(
            model_name = config["model_name"], 
            temperature = config["temperature"]
        )

    def llm_chain(self, config, prompt):
        llm_chain = LLMChain(
            llm = self.get_llm(config), 
            prompt = self.LLM_CHAIN_PROMPT
        )
    
        with get_openai_callback() as callback:
            completion = llm_chain.generate([{"question": prompt}])
    
        return completion, llm_chain, callback

    def rag_chain(self, config, prompt):
        #vector_store = self.get_vector_store_chroma()
        vector_store = self.get_vector_store_mongodb()

        rag_chain = RetrievalQA.from_chain_type(
            self.get_llm(config), 
            chain_type_kwargs = {"prompt": self.RAG_CHAIN_PROMPT}, 
            retriever = vector_store.as_retriever(search_kwargs = {"k": config["k"]}), 
            return_source_documents = True
        )
    
        with get_openai_callback() as callback:
            completion = rag_chain({"query": prompt})

        return completion, rag_chain, callback
