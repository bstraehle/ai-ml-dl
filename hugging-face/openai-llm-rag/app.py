import gradio as gr
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

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

#openai.api_key = os.environ["OPENAI_API_KEY"]

MONGODB_URI = os.environ["MONGODB_ATLAS_CLUSTER_URI"]
client = MongoClient(MONGODB_URI)
MONGODB_DB_NAME = "langchain_db"
MONGODB_COLLECTION_NAME = "gpt-4"
MONGODB_COLLECTION = client[MONGODB_DB_NAME][MONGODB_COLLECTION_NAME]
MONGODB_INDEX_NAME = "default"

template = """If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say 
              "🧠 Thanks for using the app - Bernd" at the end of the answer. """

llm_template = "Answer the question at the end. " + template + "Question: {question} Helpful Answer: "
rag_template = "Use the following pieces of context to answer the question at the end. " + template + "{context} Question: {question} Helpful Answer: "

LLM_CHAIN_PROMPT = PromptTemplate(input_variables = ["question"], 
                                  template = llm_template)
RAG_CHAIN_PROMPT = PromptTemplate(input_variables = ["context", "question"], 
                                  template = rag_template)

CHROMA_DIR  = "/data/chroma"
YOUTUBE_DIR = "/data/youtube"

PDF_URL       = "https://arxiv.org/pdf/2303.08774.pdf"
WEB_URL       = "https://openai.com/research/gpt-4"
YOUTUBE_URL_1 = "https://www.youtube.com/watch?v=--khbXchTeE"
YOUTUBE_URL_2 = "https://www.youtube.com/watch?v=hdhZwyf24mE"
YOUTUBE_URL_3 = "https://www.youtube.com/watch?v=vw-KWfKwvTQ"

MODEL_NAME  = "gpt-4"

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_overlap = 150,
                                                   chunk_size = 1500)
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
    llm_chain = LLMChain(llm = llm, prompt = LLM_CHAIN_PROMPT)
    result = llm_chain.run({"question": prompt})
    return result

def rag_chain(llm, prompt, db):
    rag_chain = RetrievalQA.from_chain_type(llm, 
                                            chain_type_kwargs = {"prompt": RAG_CHAIN_PROMPT}, 
                                            retriever = db.as_retriever(search_kwargs = {"k": 3}), 
                                            return_source_documents = True)
    result = rag_chain({"query": prompt})
    return result["result"]

def invoke(openai_api_key, rag_option, prompt):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (rag_option is None):
        raise gr.Error("Retrieval Augmented Generation is required.")
    if (prompt == ""):
        raise gr.Error("Prompt is required.")
    try:
        llm = ChatOpenAI(model_name = MODEL_NAME, 
                         openai_api_key = openai_api_key, 
                         temperature = 0)
        if (rag_option == "Chroma"):
            #splits = document_loading_splitting()
            #document_storage_chroma(splits)
            db = document_retrieval_chroma(llm, prompt)
            result = rag_chain(llm, prompt, db)
        elif (rag_option == "MongoDB"):
            #splits = document_loading_splitting()
            #document_storage_mongodb(splits)
            db = document_retrieval_mongodb(llm, prompt)
            result = rag_chain(llm, prompt, db)
        else:
            result = llm_chain(llm, prompt)
    except Exception as e:
        raise gr.Error(e)
    return result

description = """<strong>Overview:</strong> Reasoning application that demonstrates a <strong>Large Language Model (LLM)</strong> with 
                 <strong>Retrieval Augmented Generation (RAG)</strong> on <strong>external data</strong>.\n\n
                 <strong>Instructions:</strong> Enter an OpenAI API key and perform text generation use cases on <a href='""" + YOUTUBE_URL_1 + """'>YouTube</a>, 
                 <a href='""" + PDF_URL + """'>PDF</a>, and <a href='""" + WEB_URL + """'>Web</a> data published after LLM knowledge cutoff (example: GPT-4 data).
                 <ul style="list-style-type:square;">
                 <li>Set "Retrieval Augmented Generation" to "<strong>Off</strong>" and submit prompt "What is GPT-4?" The <strong>LLM without RAG</strong> does not know the answer.</li>
                 <li>Set "Retrieval Augmented Generation" to "<strong>Chroma</strong>" or "<strong>MongoDB</strong>" and experiment with prompts. The <strong>LLM with RAG</strong> knows the answer:</li>
                 <ol>
                 <li>What are GPT-4's media capabilities in 5 emojis and 1 sentence?</li>
                 <li>List GPT-4's exam scores and benchmark results.</li>
                 <li>Compare GPT-4 to GPT-3.5 in markdown table format.</li>
                 <li>Write a Python program that calls the GPT-4 API.</li>
                 <li>What is the GPT-4 API's cost and rate limit? Answer in English, Arabic, Chinese, Hindi, and Russian in JSON format.</li>
                 </ol>
                 </ul>\n\n
                 <strong>Technology:</strong> <a href='https://www.gradio.app/'>Gradio</a> UI using the <a href='https://openai.com/'>OpenAI</a> API and 
                 AI-native <a href='https://www.trychroma.com/'>Chroma</a> embedding database or 
                 <a href='https://www.mongodb.com/blog/post/introducing-atlas-vector-search-build-intelligent-applications-semantic-search-ai'>MongoDB</a> vector search. 
                 <strong>Speech-to-text</strong> via <a href='https://openai.com/research/whisper'>whisper-1</a> model, <strong>text embedding</strong> via 
                 <a href='https://openai.com/blog/new-and-improved-embedding-model'>text-embedding-ada-002</a> model, and <strong>text generation</strong> via 
                 <a href='""" + WEB_URL + """'>gpt-4</a> model. Implementation via AI-first <a href='https://www.langchain.com/'>LangChain</a> toolkit. 
                 In addition to the OpenAI API version, see also the <a href='https://aws.amazon.com/bedrock/'>Amazon Bedrock</a> API and 
                 <a href='https://cloud.google.com/vertex-ai'>Google Vertex AI</a> API versions on 
                 <a href='https://github.com/bstraehle/ai-ml-dl/tree/main/hugging-face'>GitHub</a>."""

gr.close_all()
demo = gr.Interface(fn=invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", value = "sk-", lines = 1), 
                              gr.Radio(["Off", "Chroma", "MongoDB"], label="Retrieval Augmented Generation", value = "Off"),
                              gr.Textbox(label = "Prompt", value = "What is GPT-4?", lines = 1)],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    title = "Generative AI - LLM & RAG",
                    description = description)
demo.launch()
