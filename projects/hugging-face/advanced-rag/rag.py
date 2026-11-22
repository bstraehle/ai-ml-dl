import logging
import os
import time
from collections.abc import Sequence
from functools import cache

from openai import OpenAI
from pymongo import MongoClient
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

DB_NAME = "airbnb_dataset"
COLLECTION_NAME = "listings_reviews"
VECTOR_INDEX = "vector_index"

CHAT_MODEL = "gpt-6-luna"
REASONING_EFFORT = "low"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_PATH = "description_embedding"
NUM_CANDIDATES = 150
LIMIT = 25

SYSTEM_PROMPT = "You are an AirBnB listing recommendation system."
NO_RESULTS = "No results found."
PICTURE_NOTE = "Show listing pictures as Markdown images, e.g. ![listing name](picture_url), not as links.\n"
RANKED_CONTEXT_NOTE = (
    "The listings above are ordered best first by weightedAverageReview, a review score that accounts "
    "for both rating and number of reviews. Prefer higher-ranked listings, and for each listing you "
    "recommend, state its review_scores_rating and number_of_reviews.\n"
)

MIN_REVIEWS = 3
PRIOR_AVERAGE_REVIEW = 4.77  # mean review score across reviewed listings
PRIOR_REVIEW_COUNT = 10      # weight of the prior, in reviews (median listing has 15)

# Fields sent to the LLM as context
INCLUDE_FIELDS = (
    "name", "listing_url", "picture_url", "neighbourhood_cleansed", "description",
    "neighborhood_overview", "property_type", "room_type", "accommodates", "bedrooms", "beds",
    "bathrooms_text", "amenities", "price", "review_scores_rating", "number_of_reviews",
    "weightedAverageReview",
)

REVIEW_SCORE_FIELDS = (
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication", "review_scores_location",
    "review_scores_value",
)


@cache
def get_collection() -> Collection:
    client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"], appname="advanced-rag")
    return client[DB_NAME][COLLECTION_NAME]


def get_text_embedding(client: OpenAI, text: str) -> list[float]:
    return client.embeddings.create(
        input=text, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS
    ).data[0].embedding


def vector_search(client: OpenAI, prompt: str, index_filter: dict | None = None,
                  additional_stages: Sequence[dict] = ()) -> list[dict]:
    vector_search_stage = {
        "index": VECTOR_INDEX,
        "queryVector": get_text_embedding(client, prompt),
        "path": EMBEDDING_PATH,
        "numCandidates": NUM_CANDIDATES,
        "limit": LIMIT,
    }
    if index_filter:
        vector_search_stage["filter"] = index_filter

    pipeline = [
        {"$vectorSearch": vector_search_stage},
        *additional_stages,
        {"$project": {"_id": 0, **dict.fromkeys(INCLUDE_FIELDS, 1)}},
    ]

    start = time.perf_counter()
    results = list(get_collection().aggregate(pipeline))
    logger.info("Vector search: %d results in %.0f ms", len(results), (time.perf_counter() - start) * 1000)
    return results


def get_stage_filter_result(accommodates: int, bedrooms: int) -> dict:
    return {"$match": {"accommodates": {"$eq": accommodates}, "bedrooms": {"$eq": bedrooms}}}


def get_stage_average_review_and_review_count() -> dict:
    return {
        "$addFields": {
            "averageReview": {"$avg": [f"${field}" for field in REVIEW_SCORE_FIELDS]},
            "reviewCount": "$number_of_reviews",
        }
    }


def get_stage_weighting() -> dict:
    # Bayesian average: shrink each listing's score toward the dataset mean in
    # proportion to how few reviews it has, keeping the result on the 0-5 scale.
    return {
        "$addFields": {
            "weightedAverageReview": {
                "$divide": [
                    {"$add": [
                        {"$multiply": ["$averageReview", "$reviewCount"]},
                        PRIOR_AVERAGE_REVIEW * PRIOR_REVIEW_COUNT,
                    ]},
                    {"$add": ["$reviewCount", PRIOR_REVIEW_COUNT]},
                ]
            }
        }
    }


def get_stage_sorting() -> dict:
    return {"$sort": {"weightedAverageReview": -1}}


def rag_retrieval_naive(client: OpenAI, prompt: str, accommodates: int, bedrooms: int) -> list[dict]:
    # Naive RAG: Semantic search
    return vector_search(
        client, prompt, additional_stages=[get_stage_filter_result(accommodates, bedrooms)]
    )


def rag_retrieval_advanced(client: OpenAI, prompt: str, accommodates: int, bedrooms: int) -> list[dict]:
    # Advanced RAG: Semantic search plus...

    # 1a) Pre-retrieval processing: index filter (accomodates, bedrooms, min reviews) plus...
    # 1b) Post-retrieval processing: result filter (accomodates, bedrooms) plus...
    # 2)  Bayesian weighted average review, sorted in descending order
    index_filter = {
        "$and": [
            {"accommodates": {"$eq": accommodates}},
            {"bedrooms": {"$eq": bedrooms}},
            {"number_of_reviews": {"$gte": MIN_REVIEWS}},
        ]
    }
    additional_stages = [
        get_stage_average_review_and_review_count(),
        get_stage_weighting(),
        get_stage_sorting(),
    ]
    return vector_search(client, prompt, index_filter, additional_stages)


def build_prompt(prompt: str, context: str | None = None, ranked: bool = False) -> str:
    instructions = (
        "Use the following pieces of context to answer the question at the end.\n"
        if context is not None
        else "Answer the question.\n"
    )
    return (
        instructions
        + "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
        "Keep the answer as concise as possible.\n\n"
        + (f"{context}\n\n{PICTURE_NOTE}" if context is not None else "")
        + (f"{RANKED_CONTEXT_NOTE}\n" if ranked else "")
        + f"Question: {prompt}\n"
        "Helpful Answer: "
    )


def invoke_llm(client: OpenAI, content: str) -> str:
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        reasoning_effort=REASONING_EFFORT,
    )
    return completion.choices[0].message.content


def inference(client: OpenAI, prompt: str) -> str:
    return invoke_llm(client, build_prompt(prompt))


def rag_inference(client: OpenAI, prompt: str, retrieval_result: list[dict], ranked: bool = False) -> str:
    context = str(retrieval_result) if retrieval_result else NO_RESULTS
    return invoke_llm(client, build_prompt(prompt, context, ranked and bool(retrieval_result)))
