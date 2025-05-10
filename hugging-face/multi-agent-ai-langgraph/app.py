import gradio as gr
import os, threading

from multi_agent import run_multi_agent

lock = threading.Lock()

os.environ["LANGCHAIN_PROJECT"] = "langgraph-multi-agent"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

LLM = "gpt-4.1"

def invoke(openai_api_key, topic):
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")
    if not topic:
        raise gr.Error("Topic is required.")
        
    with lock:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        article = run_multi_agent(LLM, topic)
        del os.environ["OPENAI_API_KEY"]
        return article

def clear():
    return ""

gr.close_all()

with gr.Blocks() as assistant:
    gr.Markdown("## Multi-Agent AI: Article Writing")
    gr.Markdown(os.environ.get("DESCRIPTION"))

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                openai_api_key = gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1)
                topic = gr.Textbox(label = "Topic", value=os.environ["TOPIC"], lines = 1)            
            with gr.Row():
                clear_btn = gr.ClearButton(
                    components=[openai_api_key, topic]
                )
                submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=3):
            article = gr.Markdown(label = "Article", value=os.environ["OUTPUT"], line_breaks = True, sanitize_html = False)

    clear_btn.click(
        fn=clear,
        outputs=article
    )
    
    submit_btn.click(
        fn=invoke,
        inputs=[openai_api_key, topic],
        outputs=article
    )

assistant.launch()
