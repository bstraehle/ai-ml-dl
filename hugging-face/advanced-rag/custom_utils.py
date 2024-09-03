import openai, os, time

from datasets import load_dataset
from pymongo.mongo_client import MongoClient

DB_NAME = "airbnb_dataset"
COLLECTION_NAME = "listings_reviews"

def connect_to_database():
    MONGODB_ATLAS_CLUSTER_URI = os.environ["MONGODB_ATLAS_CLUSTER_URI"]
    mongo_client = MongoClient(MONGODB_ATLAS_CLUSTER_URI, appname="advanced-rag")
    db = mongo_client.get_database(DB_NAME)
    collection = db.get_collection(COLLECTION_NAME)
    return db, collection

def rag_ingestion(collection):
    dataset = load_dataset("bstraehle/airbnb-san-francisco-202403-embed", streaming=True, split="train")
    collection.delete_many({})
    collection.insert_many(dataset)
    return "Manually create a vector search index (in free tier, this feature is not available via SDK)"

def rag_retrieval_naive(openai_api_key, 
                        prompt, 
                        db, 
                        collection, 
                        vector_index="vector_index"):
    # Naive RAG: Semantic search
    retrieval_result = vector_search_naive(
        openai_api_key, 
        prompt, 
        db, 
        collection, 
        vector_index
    )

    if not retrieval_result:
        return "No results found."

    #print("###")
    #print(retrieval_result)
    #print("###")
    
    return retrieval_result

def rag_retrieval_advanced(openai_api_key, 
                           prompt, 
                           accomodates, 
                           bedrooms, 
                           db, 
                           collection, 
                           vector_index="vector_index"):
    # Advanced RAG: Semantic search plus...
    
    # 1a) Pre-retrieval processing: index filter (accomodates, bedrooms) plus...
    
    # 1b) Post-retrieval processing: result filter (accomodates, bedrooms) plus...
    
    #match_stage = {
    #    "$match": {
    #        "accommodates": { "$eq": 2},
    #        "bedrooms": { "$eq": 1}
    #    }
    #}
    
    #additional_stages = [match_stage]

    # 2) Average review score and review count boost, sorted in descending order

    review_average_stage = {
        "$addFields": {
            "averageReviewScore": {
                "$divide": [
                    {
                        "$add": [
                            "$review_scores_rating",
                            "$review_scores_accuracy",
                            "$review_scores_cleanliness",
                            "$review_scores_checkin",
                            "$review_scores_communication",
                            "$review_scores_location",
                            "$review_scores_value",
                        ]
                    },
                    7
                ]
            },
            "reviewCountBoost": "$number_of_reviews"
        }
    }

    weighting_stage = {
        "$addFields": {
            "combinedScore": {
                "$add": [
                    {"$multiply": ["$averageReviewScore", 0.9]},
                    {"$multiply": ["$reviewCountBoost", 0.1]},
                ]
            }
        }
    }

    sorting_stage_sort = {
        "$sort": {"combinedScore": -1}
    }

    additional_stages = [review_average_stage, weighting_stage, sorting_stage_sort]
    
    retrieval_result = vector_search_advanced(
        openai_api_key, 
        prompt, 
        accomodates, 
        bedrooms, 
        db, 
        collection, 
        additional_stages, 
        vector_index
    )

    if not retrieval_result:
        return "No results found."

    #print("###")
    #print(retrieval_result)
    #print("###")
    
    return retrieval_result

def inference(openai_api_key, prompt):
    content = f"Answer this user question: {prompt}"
    return invoke_llm(openai_api_key, content)

def rag_inference(openai_api_key, prompt, retrieval_result):
    content = f"Answer this user question: {prompt} with the following context:\n{retrieval_result}"
    return invoke_llm(openai_api_key, content)

def invoke_llm(openai_api_key, content):
    openai.api_key = openai_api_key

    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system", 
                "content": "You are an AirBnB listing recommendation system."},
            {
                "role": "user", 
                "content": content
            }
        ]
    )

    return completion.choices[0].message.content
    
def vector_search_naive(openai_api_key, 
                        user_query, 
                        db, 
                        collection, 
                        vector_index="vector_index"):
    query_embedding = get_text_embedding(openai_api_key, user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    vector_search_stage = {
        "$vectorSearch": {
            "index": vector_index,
            "queryVector": query_embedding,
            "path": "description_embedding",
            "numCandidates": 150,
            "limit": 25,
        }
    }

    remove_embedding_stage =     {
        "$unset": "description_embedding"
    }
    
    pipeline = [vector_search_stage, remove_embedding_stage]

    return invoke_search(collection, pipeline)

def vector_search_advanced(openai_api_key, 
                           user_query, 
                           accommodates, 
                           bedrooms, 
                           db, 
                           collection, 
                           additional_stages=[], 
                           vector_index="vector_index"):
    query_embedding = get_text_embedding(openai_api_key, user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    vector_search_stage = {
        "$vectorSearch": {
            "index": vector_index,
            "queryVector": query_embedding,
            "path": "description_embedding",
            "numCandidates": 150,
            "limit": 25,
            "filter": {
                "$and": [
                    {"accommodates": {"$eq": accommodates}}, 
                    {"bedrooms": {"$eq": bedrooms}}
                ]
            },
        }
    }

    remove_embedding_stage =     {
        "$unset": "description_embedding"
    }
    
    pipeline = [vector_search_stage, remove_embedding_stage] + additional_stages

    return invoke_search(collection, pipeline)

def invoke_search(collection, pipeline):
    results = collection.aggregate(pipeline)
    return list(results)

def get_text_embedding(openai_api_key, text):
    if not text or not isinstance(text, str):
        return None

    openai.api_key = openai_api_key
        
    try:
        return openai.embeddings.create(
            input=text,
            model="text-embedding-3-small", dimensions=1536
        ).data[0].embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None
