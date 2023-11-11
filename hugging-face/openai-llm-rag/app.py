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

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

#openai.api_key = os.environ["OPENAI_API_KEY"]

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

def invoke(openai_api_key, use_rag, prompt):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (use_rag is None):
        raise gr.Error("Retrieval Augmented Generation is required.")
    if (prompt == ""):
        raise gr.Error("Prompt is required.")
    try:
        llm = ChatOpenAI(model_name = MODEL_NAME, 
                         openai_api_key = openai_api_key, 
                         temperature = 0)
        if (use_rag):
            # Document loading
            #docs = []
            # Load PDF
            #loader = PyPDFLoader(PDF_URL)
            #docs.extend(loader.load())
            # Load Web
            #loader = WebBaseLoader(WEB_URL_1)
            #docs.extend(loader.load())
            # Load YouTube
            #loader = GenericLoader(YoutubeAudioLoader([YOUTUBE_URL_1,
            #                                           YOUTUBE_URL_2,
            #                                           YOUTUBE_URL_3], YOUTUBE_DIR), 
            #                       OpenAIWhisperParser())
            #docs.extend(loader.load())
            # Document splitting
            #text_splitter = RecursiveCharacterTextSplitter(chunk_overlap = 150,
            #                                               chunk_size = 1500)
            #splits = text_splitter.split_documents(docs)
            # Document storage
            #vector_db = Chroma.from_documents(documents = splits, 
            #                                  embedding = OpenAIEmbeddings(disallowed_special = ()), 
            #                                  persist_directory = CHROMA_DIR)
            # Document retrieval
            vector_db = Chroma(embedding_function = OpenAIEmbeddings(),
                               persist_directory = CHROMA_DIR)
            rag_chain = RetrievalQA.from_chain_type(llm, 
                                                    chain_type_kwargs = {"prompt": RAG_CHAIN_PROMPT}, 
                                                    retriever = vector_db.as_retriever(search_kwargs = {"k": 3}), 
                                                    return_source_documents = True)
            result = rag_chain({"query": prompt})
            result = result["result"]
        else:
            chain = LLMChain(llm = llm, prompt = LLM_CHAIN_PROMPT)
            result = chain.run({"question": prompt})
    except Exception as e:
        raise gr.Error(e)
    return result

description = """<strong>Overview:</strong> Reasoning application that demonstrates a <strong>Large Language Model (LLM)</strong> with 
                 <strong>Retrieval Augmented Generation (RAG)</strong> on <strong>external data</strong>.\n\n
                 <strong>Instructions:</strong> Enter an OpenAI API key and perform LLM use cases (semantic search, summarization, translation, etc.) on 
                 <a href='""" + YOUTUBE_URL_1 + """'>YouTube</a>, <a href='""" + PDF_URL + """'>PDF</a>, and <a href='""" + WEB_URL + """'>Web</a> 
                 <strong>data on GPT-4</strong> (published after LLM knowledge cutoff).
                 <ul style="list-style-type:square;">
                 <li>Set "Retrieval Augmented Generation" to "<strong>False</strong>" and submit prompt "What is GPT-4?" The LLM <strong>without</strong> RAG does not know the answer.</li>
                 <li>Set "Retrieval Augmented Generation" to "<strong>True</strong>" and submit prompt "What is GPT-4?" The LLM <strong>with</strong> RAG knows the answer.</li>
                 <li>Experiment with prompts, e.g. "What are GPT-4's media capabilities in 3 emojis and 1 sentence?", "List GPT-4's exam scores and benchmark results.", or "Compare GPT-4 to GPT-3.5 in markdown table format."</li>
                 <li>Experiment some more, for example "What is the GPT-4 API's cost and rate limit? Answer in English, Arabic, Chinese, Hindi, and Russian in JSON format." or "Write a Python program that calls the GPT-4 API."</li>
                 </ul>\n\n
                 <strong>Technology:</strong> <a href='https://www.gradio.app/'>Gradio</a> UI using <a href='https://openai.com/'>OpenAI</a> API via AI-first 
                 <a href='https://www.langchain.com/'>LangChain</a> toolkit with <a href='""" + WEB_URL + """'>GPT-4</a> foundation model and AI-native 
                 <a href='https://www.trychroma.com/'>Chroma</a> embedding database. Speech-to-text via <a href='https://openai.com/research/whisper'>Whisper</a> 
                 foundation model."""

gr.close_all()
demo = gr.Interface(fn=invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", value = "sk-", lines = 1), 
                              gr.Radio([True, False], label="Retrieval Augmented Generation", value = False), 
                              gr.Textbox(label = "Prompt", value = "What is GPT-4?", lines = 1)],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    title = "Generative AI - LLM & RAG",
                    description = description)
demo.launch()
