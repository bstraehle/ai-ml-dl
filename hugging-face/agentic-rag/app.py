import gradio as gr
import agentops, os

from crew import get_crew

LLM = "gpt-4o"

def invoke(openai_api_key, topic, word_count=1000):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (topic == ""):
        raise gr.Error("Topic is required.")
        
    agentops.init(os.environ["AGENTOPS_API_KEY"])

    os.environ["OPENAI_API_KEY"] = openai_api_key

    result = get_crew(LLM).kickoff(inputs={"topic": topic, "word_count": word_count})

    return result

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1),
                              gr.Textbox(label = "Topic", value=os.environ["TOPIC"], lines = 1),
                              gr.Number(label = "Word Count", value=1000, minimum=500, maximum=5000)],
                    outputs = [gr.Markdown(label = "Generated Blog Post", value=os.environ["OUTPUT"])],
                    title = "Multi-Agent RAG: Blog Post Generation",
                    description = os.environ["DESCRIPTION"])

demo.launch()
