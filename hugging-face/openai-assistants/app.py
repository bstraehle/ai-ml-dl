import gradio as gr
import json, openai, os, time

from openai import OpenAI

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

_assistant, _thread = None, None

def create_assistant(client):
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
        model="gpt-4-1106-preview",
    )
    show_json("assistant", assistant)
    return assistant
    
def create_thread(client):
    thread = client.beta.threads.create()
    show_json("thread", thread)
    return thread

def create_message(client, thread, msg):
    message = client.beta.threads.messages.create(
        role="user",
        thread_id=thread.id,
        content=msg,
    )
    show_json("message", message)
    return message

def create_run(client, assistant, thread):
    run = client.beta.threads.runs.create(
        assistant_id=assistant.id,
        thread_id=thread.id,
    )
    show_json("run", run)
    return run

def wait_on_run(client, thread, run):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.25)
    show_json("run", run)
    return run

def list_messages(client, thread):
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    show_json("messages", messages)
    return messages
    
def extract_content_values(data):
    content_values = []
    for item in data.data:
        for content in item.content:
            if content.type == 'text':
                content_values.append(content.text.value)
    return content_values

def show_json(str, obj):
    print(f"=> {str}\n{json.loads(obj.model_dump_json())}")

def chat(message, history, openai_api_key):
    global _client, _assistant, _thread     
       
    if _assistant == None:
        _assistant = create_assistant(_client)

    if _thread == None:
        _thread = create_thread(_client)
        
    create_message(_client, _thread, message)

    # async
    run = create_run(_client, _assistant, _thread)
    run = wait_on_run(_client, _thread, run)
    
    messages = list_messages(_client, _thread)

    return extract_content_values(messages)[0]
        
gr.ChatInterface(
    chat,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Question", container=False, scale=7),
    title="Multi-Assistant Demo",
    description="Ask AAA Assistant, BBB Assistant, and CCC Assistant any question",
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
    examples=[["I need to solve the equation '2x + 10 = 7.5'. Can you help me?", "sk-<BringYourOwn>"]],
    cache_examples=False,
    additional_inputs=[
        gr.Textbox("sk-", label="OpenAI API Key", type = "password"),
    ],
).launch()
