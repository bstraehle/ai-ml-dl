# TODO:
#
# 1. Gradio session / multi-user thread
# 2. Function calling - https://platform.openai.com/docs/assistants/tools/function-calling
#    - get_stock_price - yfinance

# Reference:
#
# https://vimeo.com/990334325/56b552bc7a
# https://platform.openai.com/playground/assistants
# https://cookbook.openai.com/examples/assistants_api_overview_python
# https://platform.openai.com/docs/api-reference/assistants/createAssistant
# https://platform.openai.com/docs/assistants/tools

import gradio as gr
import json, openai, os, time

from datetime import date
from openai import OpenAI
from utils import function_to_schema, show_json

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

assistant, thread = None, None

def today_tool(text: str) -> str:
    """Returns today's date. Use this for any questions related to knowing today's date. 
       The input should always be an empty string, and this function will always return today's date. 
       Any date mathematics should occur outside this function."""
    return str(date.today())

tools = {
    "today_tool": today_tool,
}

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
            {"type": "function", "function": function_to_schema(today_tool)},
        ],
    )
    
    #show_json("assistant", assistant)
    
    return assistant

def load_assistant(client):
    ASSISTANT_ID = "asst_TpZgBd2QYaxUxCwUy8J9m3Bq"
    
    assistant = client.beta.assistants.retrieve(ASSISTANT_ID)

    #show_json("assistant", assistant)
    
    return assistant

def create_thread(client):
    thread = client.beta.threads.create()
    
    #show_json("thread", thread)
    
    return thread

def create_message(client, thread, msg):        
    message = client.beta.threads.messages.create(
        role="user",
        thread_id=thread.id,
        content=msg,
    )
    
    #show_json("message", message)
    
    return message

def create_run(client, assistant, thread):
    run = client.beta.threads.runs.create(
        assistant_id=assistant.id,
        thread_id=thread.id,
    )
    
    #show_json("run", run)
    
    return run

def wait_on_run(client, thread, run):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
    
        time.sleep(0.25)
    
    #show_json("run", run)
    
    return run

def get_run_steps(client, thread, run):
    run_steps = client.beta.threads.runs.steps.list(
        thread_id=thread.id,
        run_id=run.id,
        order="asc",
    )

    #show_json("run_steps", run_steps)
    
    return run_steps

def execute_tool_call(tool_call):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    return tools[name](**args)

def execute_tool_calls(run_steps):
    run_step_details = []

    tool_call_id = ""
    tool_call_result = ""
    
    for step in run_steps.data:
        step_details = step.step_details
        run_step_details.append(step_details)
        show_json("step_details", step_details)
        
        if hasattr(step_details, "tool_calls"):
            for tool_call in step_details.tool_calls:
                show_json("tool_call", tool_call)
                
                if hasattr(tool_call, "function"):
                    tool_call_id = tool_call.id
                    tool_call_result = execute_tool_call(tool_call)

    return tool_call_id, tool_call_result

def get_messages(client, thread):
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    
    #show_json("messages", messages)
    
    return messages
                        
def extract_content_values(data):
    text_values, image_values = [], []
    
    for item in data.data:
        for content in item.content:
            if content.type == "text":
                text_value = content.text.value
                text_values.append(text_value)
            if content.type == "image_file":
                image_value = content.image_file.file_id
                image_values.append(image_value)
    
    return text_values, image_values

def chat(message, history):
    if not message:
        raise gr.Error("Message is required.")
    
    global client, assistant, thread     
    
    if assistant == None:
        assistant = create_assistant(client)
    
    if thread == None or len(history) == 0:
        thread = create_thread(client)
        
    create_message(client, thread, message)

    run = create_run(client, assistant, thread)
    run = wait_on_run(client, thread, run)

    run_steps = get_run_steps(client, thread, run)
    
    tool_call_id, tool_call_result = execute_tool_calls(run_steps)

    ### TODO
    if tool_call_result:
        print("### tool_call_id=" + tool_call_id)
        print("### tool_call_result=" + tool_call_result)

        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=[
                {
                  "tool_call_id": tool_call_id,
                  "output": tool_call_result
                }
            ]
        )
    
        run = wait_on_run(client, thread, run)
        run_steps = get_run_steps(client, thread, run)
        tool_call_id, tool_call_result = execute_tool_calls(run_steps)
    ###
    
    messages = get_messages(client, thread)

    text_values, image_values = extract_content_values(messages)

    download_link = ""
    
    if len(image_values) > 0:
        download_link = f"<p>Download: https://platform.openai.com/storage/files/{image_values[0]}</p>"
    
    return f"{text_values[0]}{download_link}"

gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(height=350),
        textbox=gr.Textbox(placeholder="Ask anything", container=False, scale=7),
        title="Python Code Generator",
        description="The assistant can generate, explain, fix, optimize, document, and test code. It can also execute code.",
        clear_btn="Clear",
        retry_btn=None,
        undo_btn=None,
        examples=[
                  ["Generate: Python code to fine-tune model meta-llama/Meta-Llama-3.1-8B on dataset gretelai/synthetic_text_to_sql using QLoRA"],
                  ["Explain: r\"^(?=.*[A-Z])(?=.*[a-z])(?=.*[0-9])(?=.*[\\W]).{8,}$\""],
                  ["Fix: x = [5, 2, 1, 3, 4]; print(x.sort())"],
                  ["Optimize: x = []; for i in range(0, 10000): x.append(i)"],
                  ["Execute: First 25 Fibbonaci numbers"],
                  ["Get today's date, then generate a chart using mock data, showing stock gain YTD for NVDA, MSFT, AAPL, and GOOG, x-axis is 'Day' and y-axis is 'YTD Gain %'"]
                 ],
        cache_examples=False,
    ).launch()
