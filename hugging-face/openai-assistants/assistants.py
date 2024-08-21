import pandas as pd
import yfinance as yf

import json, openai, os, time

from datetime import date
from openai import OpenAI
from tavily import TavilyClient
from typing import List
from utils import function_to_schema, show_json

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

assistant_id = "asst_DbCpNsJ0vHSSdl6ePlkKZ8wG"

assistant, thread = None, None

def today_tool() -> str:
    """Returns today's date. Use this function for any questions related to knowing today's date. 
       There should be no input. This function always returns today's date."""
    return str(date.today())

def yf_download_tool(tickers: List[str], start_date: date, end_date: date) -> pd.DataFrame:
    """Returns historical stock data for a list of given tickers from start date to end date 
       using the yfinance library download function. 
       Use this function for any questions related to getting historical stock data. 
       The input should be the tickers as a List of strings, a start date, and an end date. 
       This function always returns a pandas DataFrame."""
    return yf.download(tickers, start=start_date, end=end_date)

def tavily_search_tool(query: str) -> str:
    """Searches the web for a given query and returns an answer, "
       ready for use as context in a RAG application, using the Tavily API. 
       Use this function for any questions requiring knowledge not available to the model. 
       The input should be the query string. This function always returns an answer string."""
    return tavily_client.get_search_context(query=query, max_results=5)

tools = {
    "today_tool": today_tool,
    "yf_download_tool": yf_download_tool,
    "tavily_search_tool": tavily_search_tool,
}

def create_assistant():
    assistant = openai_client.beta.assistants.create(
        name="Python Coding Assistant",
        instructions=(
             "You are a Python programming language expert that "
             "generates Pylint-compliant code and explains it. "
             "Execute code when explicitly asked to."
        ),
        model="gpt-4o",
        tools=[
            {"type": "code_interpreter"},
            {"type": "function", "function": function_to_schema(today_tool)},
            {"type": "function", "function": function_to_schema(yf_download_tool)},
            {"type": "function", "function": function_to_schema(tavily_search_tool)},
        ],
    )
    
    show_json("assistant", assistant)
    
    return assistant

def load_assistant():   
    assistant = openai_client.beta.assistants.retrieve(assistant_id)
    show_json("assistant", assistant)
    return assistant

def create_thread():
    thread = openai_client.beta.threads.create()
    show_json("thread", thread)
    return thread

def create_message(thread, msg):        
    message = openai_client.beta.threads.messages.create(
        role="user",
        thread_id=thread.id,
        content=msg,
    )
    
    show_json("message", message)
    return message

def create_run(assistant, thread):
    run = openai_client.beta.threads.runs.create(
        assistant_id=assistant.id,
        thread_id=thread.id,
        parallel_tool_calls=False,
    )
    
    show_json("run", run)
    return run

def wait_on_run(thread, run):
    while run.status == "queued" or run.status == "in_progress":
        run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
            
        time.sleep(1)
    
    show_json("run", run)

    if hasattr(run, "last_error") and run.last_error:
        raise gr.Error(run.last_error)

    return run

def get_run_steps(thread, run):
    run_steps = openai_client.beta.threads.runs.steps.list(
        thread_id=thread.id,
        run_id=run.id,
        order="asc",
    )

    show_json("run_steps", run_steps)
    return run_steps

def execute_tool_call(tool_call):
    name = tool_call.function.name
    args = {}

    if len(tool_call.function.arguments) > 10:
        args_json = ""

        try:
            args_json = tool_call.function.arguments
            args = json.loads(args_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing function name '{name}' function args '{args_json}': {e}")

    return tools[name](**args)

def execute_tool_calls(run_steps):
    run_step_details = []
    tool_call_ids = []
    tool_call_results = []
    
    for step in run_steps.data:
        step_details = step.step_details
        run_step_details.append(step_details)
        show_json("step_details", step_details)
        
        if hasattr(step_details, "tool_calls"):
            for tool_call in step_details.tool_calls:
                show_json("tool_call", tool_call)
                
                if hasattr(tool_call, "function"):
                    tool_call_ids.append(tool_call.id)
                    tool_call_results.append(execute_tool_call(tool_call))

    return tool_call_ids, tool_call_results

def recurse_execute_tool_calls(thread, run, run_steps, iteration):
    tool_call_ids, tool_call_results = execute_tool_calls(run_steps)
    
    if len(tool_call_ids) > iteration:
        tool_output = {}
        
        try:
            tool_output = {
                "tool_call_id": tool_call_ids[iteration],
                "output": tool_call_results[iteration].to_json()
            }
        except AttributeError:
            tool_output = {
                "tool_call_id": tool_call_ids[iteration],
                "output": tool_call_results[iteration]
            }
        
        # https://platform.openai.com/docs/api-reference/runs/submitToolOutputs
        run = openai_client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=[tool_output]
        )
    
        run = wait_on_run(thread, run)
        run_steps = get_run_steps(thread, run)
        recurse_execute_tool_calls(thread, run, run_steps, iteration + 1)
    else:
        return

def get_messages(thread):
    messages = openai_client.beta.threads.messages.list(
        thread_id=thread.id
    )
    
    show_json("messages", messages)
    return messages
                        
def extract_content_values(data):
    text_values, image_values = [], []
    
    for item in data.data:
        for content in item.content:
            # TODO: Handle other file types
            if content.type == "text":
                text_value = content.text.value
                text_values.append(text_value)
            if content.type == "image_file":
                image_value = content.image_file.file_id
                image_values.append(image_value)
    
    return text_values, image_values
