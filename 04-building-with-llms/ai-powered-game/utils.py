# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv

# these expect to find a .env file at the directory above the lesson.                                                         # the format for that file is (without the comment)                                                                           # API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService                                                                                                                                     
def load_env():
    _ = load_dotenv(find_dotenv())

def save_world(world, filename):
    with open(filename, 'w') as f:
        json.dump(world, f)

def load_world(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_together_api_key():
    load_env()
    together_api_key = os.getenv("TOGETHER_API_KEY")
    return together_api_key

def run_action(message, history, model):
    
    if(message == 'start game'):
        return start
    system_prompt = """You are an AI Game master. Your job is to write what \
happens next in a player's adventure game.\
Instructions: \
Write only 3-5 sentences in response. \
Always write in second person present tense. \
Ex. (You look north and see...)"""
    
    world_info = f"""
World: {world}
Kingdom: {kingdom}
Town: {town}
Your Character:  {character}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": world_info}
    ]

    for action in history:
        messages.append({"role": "assistant", "content": action[0]})
        messages.append({"role": "user", "content": action[1]})
           
    messages.append({"role": "user", "content": message})
    model_output = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    result = model_output.choices[0].message.content
    return result
    
def main_loop(message, history, model="meta-llama/Llama-3-70b-chat-hf"):
    return run_action(message, history, model)