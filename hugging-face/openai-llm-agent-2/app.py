import gradio as gr
import os

from agent import invoke_agent
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

config = {
    "model": "gpt-4-0613",
    "temperature": 0
}

AGENT_OFF = False
AGENT_ON  = True

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
                temperature = config["temperature"])

            output = completion.choices[0].message.content
        else:
            output = invoke_agent(
                config["model"],
                config["temperature"],
                openai_api_key,
                prompt)
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
