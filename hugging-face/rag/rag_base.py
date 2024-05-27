import os

class BaseRAG:
    PDF_URL       = "https://arxiv.org/pdf/2303.08774.pdf"
    WEB_URL       = "https://openai.com/research/gpt-4"
    YOUTUBE_URL_1 = "https://www.youtube.com/watch?v=--khbXchTeE"
    YOUTUBE_URL_2 = "https://www.youtube.com/watch?v=hdhZwyf24mE"

    MONGODB_ATLAS_CLUSTER_URI = os.environ["MONGODB_ATLAS_CLUSTER_URI"]
    MONGODB_COLLECTION_NAME   = "gpt-4"
    MONGODB_INDEX_NAME        = "default"
