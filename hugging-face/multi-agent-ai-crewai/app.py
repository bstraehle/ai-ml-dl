import gradio as gr
import agentops, os, threading

from crew import get_crew

lock = threading.Lock()

LLM = "gpt-4o"
VERBOSE = False

def invoke(openai_api_key, topic):
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")
    if not topic:
        raise gr.Error("Topic is required.")
        
    agentops.init(os.environ["AGENTOPS_API_KEY"])

    with lock:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
        article = str(get_crew(LLM, VERBOSE).kickoff(inputs={"topic": topic}))
    
        print("===")
        print(article)
        print("===")

        del os.environ["OPENAI_API_KEY"]
    
        return article

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1),
                              gr.Textbox(label = "Topic", value=os.environ["TOPIC"], lines = 1)],
                    outputs = [gr.Markdown(label = "Article", value=os.environ["OUTPUT"], line_breaks = True, sanitize_html = False)],
                    title = "Multi-Agent AI: Article Writing",
                    description = os.environ["DESCRIPTION"])

demo.launch()
