import logging, os, sys

from llama_hub.youtube_transcript import YoutubeTranscriptReader
from llama_index import download_loader, PromptTemplate
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from pathlib import Path
from pymongo.mongo_client import MongoClient

PDF_URL       = "https://arxiv.org/pdf/2303.08774.pdf"
WEB_URL       = "https://openai.com/research/gpt-4"
YOUTUBE_URL_1 = "https://www.youtube.com/watch?v=--khbXchTeE"
YOUTUBE_URL_2 = "https://www.youtube.com/watch?v=hdhZwyf24mE"

MONGODB_ATLAS_CLUSTER_URI = os.environ["MONGODB_ATLAS_CLUSTER_URI"]
MONGODB_DB_NAME           = "llamaindex_db"
MONGODB_COLLECTION_NAME   = "gpt-4"
MONGODB_INDEX_NAME        = "default"

RAG_PROMPT = PromptTemplate(os.environ["RAG_TEMPLATE"])

logging.basicConfig(stream = sys.stdout, level = logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream = sys.stdout))

def load_documents():
    docs = []
    
    # PDF
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    out_dir = Path("data")
    
    if not out_dir.exists():
        os.makedirs(out_dir)
    
    out_path = out_dir / "gpt-4.pdf"
    
    if not out_path.exists():
        r = requests.get(PDF_URL)
        with open(out_path, "wb") as f:
            f.write(r.content)

    docs.extend(loader.load_data(file = Path(out_path)))
    #print("docs = " + str(len(docs)))
    
    # Web
    SimpleWebPageReader = download_loader("SimpleWebPageReader")
    loader = SimpleWebPageReader()
    docs.extend(loader.load_data(urls = [WEB_URL]))
    #print("docs = " + str(len(docs)))

    # YouTube
    loader = YoutubeTranscriptReader()
    docs.extend(loader.load_data(ytlinks = [YOUTUBE_URL_1,
                                            YOUTUBE_URL_2]))
    #print("docs = " + str(len(docs)))
    
    return docs

def store_documents(config, docs):
    storage_context = StorageContext.from_defaults(
        vector_store = get_vector_store())
    
    VectorStoreIndex.from_documents(
        docs,
        storage_context = storage_context
    )

def get_vector_store():
    return MongoDBAtlasVectorSearch(
        MongoClient(MONGODB_ATLAS_CLUSTER_URI),
        db_name = MONGODB_DB_NAME,
        collection_name = MONGODB_COLLECTION_NAME,
        index_name = MONGODB_INDEX_NAME
    )

def rag_ingestion(config):
    docs = load_documents()
    
    store_documents(config, docs)

def rag_retrieval(config, prompt):
    index = VectorStoreIndex.from_vector_store(
        vector_store = get_vector_store())

    query_engine = index.as_query_engine()
    
    return query_engine.query(prompt)
