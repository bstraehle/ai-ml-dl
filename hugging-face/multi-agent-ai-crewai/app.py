import gradio as gr
import agentops, os

from crew import get_crew

LLM = "gpt-4o"

def invoke(openai_api_key, topic):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (topic == ""):
        raise gr.Error("Topic is required.")
        
    agentops.init(os.environ["AGENTOPS_API_KEY"])

    os.environ["OPENAI_API_KEY"] = openai_api_key

    article = get_crew(LLM).kickoff(inputs={"topic": topic})

    print("===")
    print(article)
    print("===")

    return article

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1),
                              gr.Textbox(label = "Topic", value=os.environ["TOPIC"], lines = 1)],
                    outputs = [gr.Markdown(label = "Generated Article", value=os.environ["OUTPUT"])],
                    title = "Multi-Agent AI: Article Generation",
                    description = os.environ["DESCRIPTION"])

demo.launch()
