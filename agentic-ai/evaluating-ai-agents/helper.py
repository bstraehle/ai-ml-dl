# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv
                     
def load_env():
    _ = load_dotenv(find_dotenv(), override=True)

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key


def get_phoenix_endpoint():
    load_env()
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    return phoenix_endpoint


