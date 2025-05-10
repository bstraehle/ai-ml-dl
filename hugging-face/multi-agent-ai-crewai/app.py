import gradio as gr
import agentops, os, threading

from crew import get_crew

lock = threading.Lock()

LLM_MANAGER = "gpt-4-mini"
LLM_AGENTS = "gpt-4.1"
VERBOSE = False

def invoke(openai_api_key, topic):
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")
    if not topic:
        raise gr.Error("Topic is required.")
        
    agentops.init(os.environ["AGENTOPS_API_KEY"])

    with lock:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
        article = str(get_crew(LLM_MANAGER, LLM_AGENTS, VERBOSE).kickoff(inputs={"topic": topic}))
    
        print("===")
        print(article)
        print("===")

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
