import os

from llama_hub.youtube_transcript import YoutubeTranscriptReader
from llama_index import download_loader, PromptTemplate
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from pathlib import Path
from pymongo import MongoClient
from rag_base import BaseRAG

class LlamaIndexRAG(BaseRAG):
    MONGODB_DB_NAME = "llamaindex_db"
    
    def load_documents(self):
        docs = []
    
        # PDF
        PDFReader = download_loader("PDFReader")
        loader = PDFReader()
        out_dir = Path("data")
    
        if not out_dir.exists():
            os.makedirs(out_dir)
    
        out_path = out_dir / "gpt-4.pdf"
    
        if not out_path.exists():
            r = requests.get(self.PDF_URL)
            with open(out_path, "wb") as f:
                f.write(r.content)

        docs.extend(loader.load_data(file = Path(out_path)))
        #print("docs = " + str(len(docs)))
    
        # Web
        SimpleWebPageReader = download_loader("SimpleWebPageReader")
        loader = SimpleWebPageReader()
        docs.extend(loader.load_data(urls = [self.WEB_URL]))
        #print("docs = " + str(len(docs)))

        # YouTube
        loader = YoutubeTranscriptReader()
        docs.extend(loader.load_data(ytlinks = [self.YOUTUBE_URL_1,
                                                self.YOUTUBE_URL_2]))
        #print("docs = " + str(len(docs)))
    
        return docs

    def store_documents(self, config, docs):
        storage_context = StorageContext.from_defaults(
            vector_store = self.get_vector_store())
    
        VectorStoreIndex.from_documents(
            docs,
            storage_context = storage_context
        )

    def get_vector_store(self):
        return MongoDBAtlasVectorSearch(
            MongoClient(self.MONGODB_ATLAS_CLUSTER_URI),
            db_name = self.MONGODB_DB_NAME,
            collection_name = self.MONGODB_COLLECTION_NAME,
            index_name = self.MONGODB_INDEX_NAME
        )

    def ingestion(self, config):
        docs = self.load_documents()
    
        self.store_documents(config, docs)

    def retrieval(self, config, prompt):
        index = VectorStoreIndex.from_vector_store(
            vector_store = self.get_vector_store())

        query_engine = index.as_query_engine(
            similarity_top_k = config["k"]
        )
 
        return query_engine.query(prompt)
