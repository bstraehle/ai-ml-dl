from datasets import load_dataset

from rag import get_collection

DATASET = "bstraehle/airbnb-san-francisco-202403-embed"


def rag_ingestion() -> None:
    collection = get_collection()
    dataset = load_dataset(DATASET, streaming=True, split="train")
    collection.delete_many({})
    collection.insert_many(dataset)
    print("Manually create a vector search index (in free tier, this feature is not available via SDK)")


if __name__ == "__main__":
    rag_ingestion()
