# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv

# these expect to find a .env file at the directory above the lesson.
# the format for that file is (without the comment)
#API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService                                                                                                                                     
def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

def get_llama_cloud_api_key():
    load_env()
    llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    return llama_cloud_api_key

def extract_html_content(filename):
    try:
        with open(filename, 'r') as file:
            html_content = file.read()
            html_content = f""" <div style="width: 100%; height: 800px; overflow: hidden;"> {html_content} </div>"""
            return html_content
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")