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
                        accomodates,
                        bedrooms,
                        db,
                        collection,
                        vector_index="vector_index"):
    # Naive RAG: Semantic search
    
    retrieval_result = vector_search_naive(
        openai_api_key,
        prompt,
        accomodates,
        bedrooms,
        db,
        collection,
        vector_index
    )

    if not retrieval_result:
        return "No results found."

    print(retrieval_result)
    
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
    # 2)  Weighted average review, sorted in descending order

    additional_stages = [
        get_stage_average_review_and_review_count(),
        get_stage_weighting(),
        get_stage_sorting()
    ]
    
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

    print(retrieval_result)
    
    return retrieval_result

def inference(openai_api_key, prompt):
    content = (
        "Answer the question.\n"
        "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
        "Keep the answer as concise as possible.\n\n"
        f"Question: {prompt}\n"
        "Helpful Answer: "
    )

    return invoke_llm(openai_api_key, content)

def rag_inference(openai_api_key, prompt, retrieval_result):
    content = (
        "Use the following pieces of context to answer the question at the end.\n"
        "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
        "Keep the answer as concise as possible.\n\n"
        f"{retrieval_result}\n\n"
        f"Question: {prompt}\n"
        "Helpful Answer: "
    )

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
        ],
        temperature=0.0
    )

    return completion.choices[0].message.content
    
def vector_search_naive(openai_api_key, 
                        prompt,
                        accomodates,
                        bedrooms,
                        db, 
                        collection, 
                        vector_index="vector_index"):
    query_embedding = get_text_embedding(openai_api_key, prompt)

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

    pipeline = [
        vector_search_stage, 
        get_stage_include_fields(), 
        get_stage_filter_result(accomodates, bedrooms)
    ]

    return invoke_search(db, collection, pipeline)

def vector_search_advanced(openai_api_key, 
                           prompt, 
                           accommodates, 
                           bedrooms, 
                           db, 
                           collection, 
                           additional_stages=[], 
                           vector_index="vector_index"):
    query_embedding = get_text_embedding(openai_api_key, prompt)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    vector_search_and_filter_stage = {
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
  
    pipeline = [
        vector_search_and_filter_stage, 
        get_stage_include_fields()
    ] + additional_stages

    return invoke_search(db, collection, pipeline)

def get_stage_exclude_fields():
    return {
        "$unset": "description_embedding"
    }
    
def get_stage_include_fields():
    return {
        "$project": {
            "id": 1, 
            "listing_url": 1, 
            "name": 1, 
            "description": 1, 
            "neighborhood_overview": 1, 
            "picture_url": 1, 
            "host_id": 1, 
            "host_url": 1, 
            "host_name": 1, 
            "host_since": 1, 
            "host_location": 1, 
            "host_about": 1, 
            "host_response_time": 1, 
            "host_response_rate": 1, 
            "host_acceptance_rate": 1, 
            "host_is_superhost": 1, 
            "host_thumbnail_url": 1, 
            "host_picture_url": 1, 
            "host_neighbourhood": 1, 
            "host_listings_count": 1, 
            "host_total_listings_count": 1, 
            "host_verifications": 1, 
            "host_has_profile_pic": 1, 
            "host_identity_verified": 1, 
            "neighbourhood": 1, 
            "neighbourhood_cleansed": 1, 
            "neighbourhood_group_cleansed": 1, 
            "latitude": 1, 
            "longitude": 1, 
            "property_type": 1, 
            "room_type": 1, 
            "accommodates": 1, 
            "bathrooms": 1, 
            "bathrooms_text": 1, 
            "bedrooms": 1, 
            "beds": 1, 
            "amenities": 1, 
            "price": 1, 
            "minimum_nights": 1, 
            "maximum_nights": 1, 
            "minimum_minimum_nights": 1, 
            "maximum_minimum_nights": 1, 
            "minimum_maximum_nights": 1, 
            "maximum_maximum_nights": 1, 
            "minimum_nights_avg_ntm": 1, 
            "maximum_nights_avg_ntm": 1, 
            "calendar_updated": 1, 
            "has_availability": 1, 
            "availability_30": 1, 
            "availability_60": 1, 
            "availability_90": 1, 
            "availability_365": 1, 
            "number_of_reviews": 1, 
            "number_of_reviews_ltm": 1, 
            "number_of_reviews_l30d": 1, 
            "first_review": 1, 
            "last_review": 1, 
            "review_scores_rating": 1, 
            "review_scores_accuracy": 1, 
            "review_scores_cleanliness": 1, 
            "review_scores_checkin": 1, 
            "review_scores_communication": 1, 
            "review_scores_location": 1, 
            "review_scores_value": 1, 
            "license": 1, 
            "instant_bookable": 1, 
            "calculated_host_listings_count": 1, 
            "calculated_host_listings_count_entire_homes": 1, 
            "calculated_host_listings_count_private_rooms": 1, 
            "calculated_host_listings_count_shared_rooms": 1, 
            "reviews_per_month": 1,
        }
    }

def get_stage_filter_result(accomodates, bedrooms):
    return {
        "$match": {
            "accommodates": { "$eq": accomodates},
            "bedrooms": { "$eq": bedrooms}
        }
    }

def get_stage_average_review_and_review_count():
    return {
        "$addFields": {
            "averageReview": {
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
            "reviewCount": "$number_of_reviews"
        }
    }

def get_stage_weighting():
    return {
        "$addFields": {
            "weightedAverageReview": {
                "$add": [
                    {"$multiply": ["$averageReview", 0.9]},
                    {"$multiply": ["$reviewCount", 0.1]},
                ]
            }
        }
    }

def get_stage_sorting():
    return {
        "$sort": {"weightedAverageReview": -1}
    }

def invoke_search(db, collection, pipeline):
    results = collection.aggregate(pipeline)
    
    print(f"Vector search millis elapsed: {get_millis_elapsed(db, collection, pipeline)}")
    
    return list(results)

def get_millis_elapsed(db, collection, pipeline):
    explain_query_execution = db.command(
        "explain", {
            "aggregate": collection.name,
            "pipeline": pipeline,
            "cursor": {}
        }, 
        verbosity="executionStats")

    explain_vector_search = explain_query_execution["stages"][0]["$vectorSearch"]
    
    return explain_vector_search["explain"]["collectStats"]["millisElapsed"]

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
