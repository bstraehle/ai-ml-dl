import gradio as gr
import os

from agent_langchain import agent_langchain
from agent_llamaindex import agent_llamaindex
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

AGENT_OFF = "Off"
AGENT_LANGCHAIN  = "LangChain"
AGENT_LLAMAINDEX = "LlamaIndex"

config = {
    "model": "gpt-4-0613",
    "temperature": 0
}

def invoke(openai_api_key, prompt, agent_option):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (prompt == ""):
        raise gr.Error("Prompt is required.")
    if (agent_option is None):
        raise gr.Error("Use Agent is required.")

    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    output = ""
    
    try:
        if (agent_option == AGENT_LANGCHAIN):
            completion = agent_langchain(
                config,
                prompt
            )
    
            output = completion["output"]
        elif (agent_option == AGENT_LLAMAINDEX):
            output = agent_llamaindex(
                config,
                prompt
            )
        else:
            client = OpenAI()
    
            completion = client.chat.completions.create(
                messages = [{"role": "user", "content": prompt}],
                model = config["model"],
                temperature = config["temperature"]
            )
    
            output = completion.choices[0].message.content
    except Exception as e:
        err_msg = e

        raise gr.Error(e)

    return output

gr.close_all()

demo = gr.Interface(
    fn = invoke, 
    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1),
              gr.Textbox(label = "Prompt", lines = 1, 
                         value = "How does current weather in San Francisco and Paris compare in metric and imperial system? Answer in JSON format and include today's date."),
              gr.Radio([AGENT_OFF, AGENT_LANGCHAIN, AGENT_LLAMAINDEX], label = "Use Agent", value = AGENT_LANGCHAIN)],
    outputs = [gr.Textbox(label = "Completion", lines = 1)],
    title = "Real-Time Reasoning Application",
    description = os.environ["DESCRIPTION"]
)

demo.launch()
