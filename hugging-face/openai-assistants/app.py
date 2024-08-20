# TODO: Gradio session / multi-user thread

# Reference:
#
# https://vimeo.com/990334325/56b552bc7a
# https://platform.openai.com/playground/assistants
# https://cookbook.openai.com/examples/assistants_api_overview_python
# https://platform.openai.com/docs/api-reference/assistants/createAssistant
# https://platform.openai.com/docs/assistants/tools

import gradio as gr

import json

from assistants import (
    openai_client,
    assistant,
    thread,
    create_assistant,
    load_assistant,
    create_thread,
    create_message,
    create_run,
    wait_on_run,
    get_run_steps,
    execute_tool_calls,
    get_messages,
    extract_content_values,
)

def chat(message, history):
    if not message:
        raise gr.Error("Message is required.")
    
    global assistant, thread     
    
    if assistant == None:
        #assistant = create_assistant(openai_client) # on first run, create assistant and update assistant_id
                                                     # see https://platform.openai.com/playground/assistants
        assistant = load_assistant(openai_client) # on subsequent runs, load assistant
    
    if thread == None or len(history) == 0:
        thread = create_thread(openai_client)
        
    create_message(openai_client, thread, message)

    run = create_run(openai_client, assistant, thread)

    run = wait_on_run(openai_client, thread, run)
    run_steps = get_run_steps(openai_client, thread, run)

    ### TODO
    tool_call_ids, tool_call_results = execute_tool_calls(run_steps)
    
    if len(tool_call_ids) > 0:
        # https://platform.openai.com/docs/api-reference/runs/submitToolOutputs
        tool_output = {}
        
        try:
            tool_output = {
                "tool_call_id": tool_call_ids[0],
                "output": tool_call_results[0].to_json()
            }
        except AttributeError:
            tool_output = {
                "tool_call_id": tool_call_ids[0],
                "output": tool_call_results[0]
            }
        
        run = openai_client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=[tool_output]
        )
    
        run = wait_on_run(openai_client, thread, run)
        run_steps = get_run_steps(openai_client, thread, run)
    ###
        tool_call_ids, tool_call_results = execute_tool_calls(run_steps)
            
        if len(tool_call_ids) > 1:
            # https://platform.openai.com/docs/api-reference/runs/submitToolOutputs
            tool_output = {}
            
            try:
                tool_output = {
                    "tool_call_id": tool_call_ids[1],
                    "output": tool_call_results[1].to_json()
                }
            except AttributeError:
                tool_output = {
                    "tool_call_id": tool_call_ids[1],
                    "output": tool_call_results[1]
                }
            
            run = openai_client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=[tool_output]
            )
        
            run = wait_on_run(openai_client, thread, run)
            run_steps = get_run_steps(openai_client, thread, run)    
    ###
    
    messages = get_messages(openai_client, thread)

    text_values, image_values = extract_content_values(messages)

    download_link = ""
    
    if len(image_values) > 0:
        download_link = f"<p>Download: https://platform.openai.com/storage/files/{image_values[0]}</p>"
    
    return f"{'<hr>'.join(list(reversed(text_values))[1:])}{download_link}"

gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(height=350),
        textbox=gr.Textbox(placeholder="Ask anything", container=False, scale=7),
        title="Python Coding Assistant",
        description=(
            "The assistant can **generate, explain, fix, optimize,** and **document Python code, "
            "create unit test cases,** and **answer general coding-related questions.** "
            "It can also **execute code**. "
            "The assistant has access to a <b>today tool</b> (get current date), to a "
            "**yfinance download tool** (get stock data), and to a "
            "**tavily search tool** (web search)."
        ),
        clear_btn="Clear",
        retry_btn=None,
        undo_btn=None,
        examples=[
                  ["Generate: Code to fine-tune model meta-llama/Meta-Llama-3.1-8B on dataset gretelai/synthetic_text_to_sql using QLoRA"],
                  ["Explain: r\"^(?=.*[A-Z])(?=.*[a-z])(?=.*[0-9])(?=.*[\\W]).{8,}$\""],
                  ["Fix: x = [5, 2, 1, 3, 4]; print(x.sort())"],
                  ["Optimize: x = []; for i in range(0, 10000): x.append(i)"],
                  ["Execute: First 25 Fibbonaci numbers"],
                  ["Execute with tools: Create a plot showing stock gain QTD for NVDA and AMD, x-axis is \"Day\" and y-axis is \"Gain %\""],
                  ["Execute with tools: Get key announcements from the latest OpenAI Dev Day"]
                 ],
        cache_examples=False,
    ).launch()
