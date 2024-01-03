import gradio as gr
import os

from datetime import date
from langchain.agents import AgentType, initialize_agent, load_tools, tool
from langchain.chat_models import ChatOpenAI
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

OPENWEATHERMAP_API_KEY = os.environ["OPENWEATHERMAP_API_KEY"]

config = {
    "model": "gpt-4-0613",
    "temperature": 0,
}

AGENT_OFF = False
AGENT_ON  = True

@tool
def date_tool(text: str) -> str:
    """Returns today's date. Use this for any questions related to knowing today's date. 
       The input should always be an empty string, and this function will always return today's date. 
       Any date mathematics should occur outside this function."""
    return str(date.today())

def invoke(openai_api_key, prompt, agent_option):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (prompt == ""):
        raise gr.Error("Prompt is required.")
    if (agent_option is None):
        raise gr.Error("Use Agent is required.")

    output = ""
    
    try:
        if (agent_option == AGENT_OFF):
            client = OpenAI(api_key = openai_api_key)
    
            completion = client.chat.completions.create(
                messages = [{"role": "user", "content": prompt}],
                model = config["model"],
                temperature = config["temperature"],)
    
            output = completion.choices[0].message.content
        else:
            llm = ChatOpenAI(
                model_name = config["model"],
                openai_api_key = openai_api_key, 
                temperature = config["temperature"])
    
            tools = load_tools(["openweathermap-api"])
            
            agent = initialize_agent(
                tools +      # built-in tools
                [date_tool], # custom tools
                llm,
                agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors = True,
                verbose = True)

            completion = agent(prompt)
    
            output = completion["output"]
    except Exception as e:
        err_msg = e

        raise gr.Error(e)

    return output

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1),
                              gr.Textbox(label = "Prompt", lines = 1, value = "How does current weather in Los Angeles, New York, and Paris compare in metric and imperial system? Answer in JSON format and include today's date."),
                              gr.Radio([AGENT_OFF, AGENT_ON], label = "Use Agent", value = AGENT_ON)],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    title = "Real-Time Reasoning Application",
                    description = os.environ["DESCRIPTION"])

demo.launch()
