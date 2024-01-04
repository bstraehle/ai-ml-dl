import openai, os, requests

from llama_hub.youtube_transcript import YoutubeTranscriptReader
from llama_index import GPTVectorStoreIndex, download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore
from llama_index.storage.storage_context import StorageContext

from pathlib import Path

PDF_URL       = "https://arxiv.org/pdf/2303.08774.pdf"
WEB_URL       = "https://openai.com/research/gpt-4"
YOUTUBE_URL_1 = "https://www.youtube.com/watch?v=--khbXchTeE"
YOUTUBE_URL_2 = "https://www.youtube.com/watch?v=hdhZwyf24mE"

MONGODB_ATLAS_CLUSTER_URI = os.environ["MONGODB_ATLAS_CLUSTER_URI"]
MONGODB_DB_NAME           = "llamaindex_db"

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

    # Web
    SimpleWebPageReader = download_loader("SimpleWebPageReader")
    loader = SimpleWebPageReader()
    
    docs.extend(loader.load_data(urls = [WEB_URL]))

    # YouTube
    loader = YoutubeTranscriptReader()
    
    docs.extend(loader.load_data(ytlinks = [YOUTUBE_URL_1, 
                                            YOUTUBE_URL_2]))
    
    return docs

def split_documents(docs):
    return SimpleNodeParser().get_nodes_from_documents(docs)

def store_documents(nodes):
    docstore = MongoDocumentStore.from_uri(
        uri = MONGODB_ATLAS_CLUSTER_URI,
        db_name = MONGODB_DB_NAME)
    
    docstore.add_documents(nodes)

def rag_ingestion():
    docs = load_documents()
    
    nodes = split_documents(docs)
    
    store_documents(nodes)

def retrieve_documents():
    docstore = MongoDocumentStore.from_uri(
        uri = MONGODB_ATLAS_CLUSTER_URI, 
        db_name = MONGODB_DB_NAME)

    index_store = MongoIndexStore.from_uri(
        uri = MONGODB_ATLAS_CLUSTER_URI, 
        db_name = MONGODB_DB_NAME)
    
    storage_context = StorageContext.from_defaults(
        docstore = docstore,
        index_store = index_store
    )

    nodes = list(docstore.docs.values())

    vector_index = GPTVectorStoreIndex(
        nodes, 
        storage_context = storage_context)
    
    return vector_index

def rag_retrieval(prompt):
    vector_index = retrieve_documents()

    return vector_index.as_query_engine().query(prompt)
