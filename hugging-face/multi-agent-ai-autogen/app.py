import gradio as gr
import os, threading

from multi_agent import run_multi_agent

lock = threading.Lock()

LLM_WHITE = "gpt-4.1"
LLM_BLACK = "gpt-4o"

def invoke(openai_api_key, num_moves = 10):
    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")

    with lock:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        result = run_multi_agent(LLM_WHITE, LLM_BLACK, num_moves)
        del os.environ["OPENAI_API_KEY"]
        print("===")
        print(result)
        print("===")
        return result

def clear():
    return ""

gr.close_all()

with gr.Blocks() as assistant:
    gr.Markdown("## Multi-Agent AI: Chess")
    gr.Markdown(os.environ.get("DESCRIPTION"))

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                openai_api_key = gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1)
                num_moves = gr.Number(label = "Number of Moves", value = 10, minimum = 1, maximum = 50)
            with gr.Row():
                clear_btn = gr.ClearButton(
                    components=[openai_api_key, num_moves]
                )
                submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=2):
            game = gr.Markdown(label = "Game", value=os.environ["OUTPUT"], line_breaks=True, sanitize_html=False)

    clear_btn.click(
        fn=clear,
        outputs=game
    )
    
    submit_btn.click(
        fn=invoke,
        inputs=[openai_api_key, num_moves],
        outputs=game
    )

assistant.launch()
