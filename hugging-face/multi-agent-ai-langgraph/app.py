import gradio as gr
import os, threading

from multi_agent import run_multi_agent

lock = threading.Lock()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-multi-agent"

LLM = "gpt-4o"

def invoke(openai_api_key, topic):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (topic == ""):
        raise gr.Error("Topic is required.")
        
    with lock:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        result = run_multi_agent(LLM, topic)
        del os.environ["OPENAI_API_KEY"]
        return result

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1),
                              gr.Textbox(label = "Topic", value=os.environ["TOPIC"], lines = 1)],
                    outputs = [gr.Markdown(label = "Generated Article", value=os.environ["OUTPUT"])],
                    title = "Multi-Agent AI: Article Generation",
                    description = os.environ["DESCRIPTION"])

demo.launch()
