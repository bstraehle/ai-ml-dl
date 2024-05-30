import gradio as gr
import os

from rag_langgraph import run_multi_agent

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-multi-agent"

LLM = "gpt-4o"

def invoke(openai_api_key, topic):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (topic == ""):
        raise gr.Error("Topic is required.")
        
    os.environ["OPENAI_API_KEY"] = openai_api_key
   
    return run_multi_agent(LLM, topic)

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1),
                              gr.Textbox(label = "Topic", value=os.environ["TOPIC"], lines = 1)],
                    outputs = [gr.Markdown(label = "Generated Article", value=os.environ["OUTPUT"])],
                    title = "Multi-Agent AI: Article Generation",
                    description = os.environ["DESCRIPTION"])

demo.launch()
