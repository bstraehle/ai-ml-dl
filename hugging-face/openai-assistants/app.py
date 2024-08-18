# Multimodal message https://platform.openai.com/docs/assistants/tools/code-interpreter/passing-files-to-code-interpreter
# File search https://platform.openai.com/docs/api-reference/messages/createMessage
# Matlplotlib chart
# Function: Tavily API
# Multi-user thread

# https://platform.openai.com/playground/assistants
# https://platform.openai.com/docs/api-reference/assistants/createAssistant
# https://platform.openai.com/docs/assistants/tools/code-interpreter
# https://cookbook.openai.com/examples/assistants_api_overview_python

import gradio as gr
import datetime, openai, os, time

from openai import OpenAI
from utils import show_json

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

assistant, thread = None, None

def create_assistant(client):
    assistant = client.beta.assistants.create(
        name="Python Code Generator",
        instructions=(
                         "You are a Python programming language expert that "
                         "generates Pylint-compliant code and explains it. "
                         "Only execute code when explicitly asked to."
                     ),
        model="gpt-4o",
        tools=[
                  {"type": "code_interpreter"},
                  {"type": "file_search"},
              ],
    )
    
    show_json("assistant", assistant)
    
    return assistant

def load_assistant(client):
    assistant = client.beta.assistants.retrieve("asst_kjO8BRHMREWBlY0LQ7WECfeD")

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
        content=msg["text"],
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

def get_run_steps(client, thread, run):
    run_steps = client.beta.threads.runs.steps.list(
        thread_id=thread.id,
        run_id=run.id,
        order="asc",
    )

    show_json("run_steps", run_steps)
    
    return run_steps

def get_run_step_details(run_steps):
    for step in run_steps.data:
        step_details = step.step_details
        
        show_json("step_details", step_details)

def get_messages(client, thread):
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    
    show_json("messages", messages)
    
    return messages
    
def extract_content_values(data):
    content_values = []
    
    for item in data.data:
        for content in item.content:
            if content.type == "text":
                content_values.append(content.text.value)
    
    return content_values

def chat(message, history):
    if not message:
        raise gr.Error("Message is required.")
    
    global client, assistant, thread     
    
    if assistant == None:
        assistant = load_assistant(client)
    
    if thread == None or len(history) == 0:
        thread = create_thread(client)
        
    create_message(client, thread, message)

    run = create_run(client, assistant, thread)
    run = wait_on_run(client, thread, run)

    run_steps = get_run_steps(client, thread, run)

    get_run_step_details(run_steps)
    
    messages = get_messages(client, thread)

    content_values = extract_content_values(messages)

    print("###")
    print(content_values[0])
    print("###")
    
    return content_values[0]

gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(height=350),
        textbox=gr.MultimodalTextbox(placeholder="Ask anything", container=False, scale=7),
        title="Python Code Generator",
        description="The assistant can generate code, explain, fix, optimize, document, test, and generally help with code. It can also execute code.",
        clear_btn="Clear",
        retry_btn=None,
        undo_btn=None,
        examples=[
                  [{"text": "Generate: Python code to fine-tune model meta-llama/Meta-Llama-3.1-8B on dataset gretelai/synthetic_text_to_sql using QLoRA", "files": []}],
                  [{"text": "Explain: r\"^(?=.*[A-Z])(?=.*[a-z])(?=.*[0-9])(?=.*[\\W]).{8,}$\"", "files": []}],
                  [{"text": "Fix: x = [5, 2, 1, 3, 4]; print(x.sort())", "files": []}],
                  [{"text": "Optimize: x = []; for i in range(0, 10000): x.append(i)", "files": []}],
                  [{"text": "Execute: First 25 Fibbonaci numbers", "files": []}],
                  [{"text": "Execute: Chart showing stock gain YTD for NVDA, MSFT, AAPL, and GOOG, x-axis is 'Day' and y-axis is 'YTD Gain %'", "files": []}],
                 ],
        cache_examples=False,
        multimodal=True,
    ).launch()
