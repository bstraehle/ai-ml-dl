# Reference:
#
# https://vimeo.com/990334325/56b552bc7a
# https://platform.openai.com/playground/assistants
# https://cookbook.openai.com/examples/assistants_api_overview_python
# https://platform.openai.com/docs/api-reference/assistants/createAssistant
# https://platform.openai.com/docs/assistants/tools

import gradio as gr

import os, threading

from assistants import (
    set_openai_client,
    get_assistant,
    set_assistant,
    get_thread,
    set_thread,
    create_assistant,
    load_assistant,
    create_thread,
    create_message,
    create_run,
    wait_on_run,
    get_run_steps,
    recurse_execute_tool_calls,
    get_messages,
    extract_content_values,
)

lock = threading.Lock()

def chat(message, history, openai_api_key):
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required (see additional inputs below).")
    if not message:
        raise gr.Error("Message is required.")

    with lock:
        result_text, result_image = "", ""
    
        try:
            if os.environ["OPENAI_API_KEY"] != openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
                
                set_openai_client()
                
                #set_assistant(create_assistant()) # first run
                set_assistant(load_assistant()) # subsequent runs
                
            if get_thread() == None or len(history) == 0:
                set_thread(create_thread())
                
            create_message(get_thread(), message)
            run = create_run(get_assistant(), get_thread())
            run = wait_on_run(get_thread(), run)
            run_steps = get_run_steps(get_thread(), run)
            recurse_execute_tool_calls(get_thread(), run, run_steps, 0)
            messages = get_messages(get_thread())
            text_values, image_values = extract_content_values(messages)
            
            for text_value in list(reversed(text_values)):
                result_text += f"{text_value}<br><br>"

            if len(image_values) > 0:
                result_image = f"[https://platform.openai.com/storage/files/{image_values[0]}](https://platform.openai.com/storage/files/{image_values[0]})"
        except Exception as e:
            raise gr.Error(e)

        return f"{result_text}{result_image}"

gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(height=250),
        textbox=gr.Textbox(placeholder="Ask anything", container=False, scale=7),
        title="Python Coding Assistant",
        description=os.environ.get("DESCRIPTION"),
        clear_btn="Clear",
        retry_btn=None,
        undo_btn=None,
        examples=[
                  ["Generate: Code to fine-tune model meta-llama/Meta-Llama-3.1-8B on dataset gretelai/synthetic_text_to_sql using QLoRA"],
                  ["Explain: r\"^(?=.*[A-Z])(?=.*[a-z])(?=.*[0-9])(?=.*[\\W]).{8,}$\""],
                  ["Fix: x = [5, 2, 1, 3, 4]; print(x.sort())"],
                  ["Optimize: x = []; for i in range(0, 10000): x.append(i)"],
                  ["1. Execute: Calculate the first 25 Fibbonaci numbers. 2. Show the code."],
                  ["1. Execute with tools: Create a plot showing stock gain QTD for NVDA and AMD, x-axis is \"Day\" and y-axis is \"Gain %\". 2. Show the code."],
                  ["1. Execute with tools: Get key announcements from latest OpenAI Dev Day. 2. Show the web references."]
                 ],
        cache_examples=False,
        additional_inputs=[
            gr.Textbox("", label="OpenAI API Key", type = "password")
        ],
    ).launch()
