import os, threading
import gradio as gr
from crew import run_crew
from utils import get_questions

QUESTION_FILE_PATH = "data/gaia_validation.jsonl"
QUESTION_LEVEL     = 1

def _run(question, openai_api_key, gemini_api_key, file_name = ""):
    """
    Run GAIA General AI Assistant to answer a question.

    Args:
        question (str): The question to answer
        openai_api_key (str): OpenAI API key
        gemini_api_key (str): Gemini API key
        file_name (str): Optional file name

    Returns:
        str: The answer to the question
    """
    if not question:
        raise gr.Error("Question is required.")

    if not openai_api_key:
        raise gr.Error("OpenAI API Key is required.")
    
    if not gemini_api_key:
        raise gr.Error("Gemini API Key is required.")

    if file_name:
        file_name = f"data/{file_name}"
    
    lock = threading.Lock()
    
    with lock:
        answer = ""

        try:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            os.environ["GEMINI_API_KEY"] = gemini_api_key
            
            answer = run_crew(question, file_name)
        except Exception as e:
            raise gr.Error(e)
        finally:
            del os.environ["OPENAI_API_KEY"]
            del os.environ["GEMINI_API_KEY"]
        
        return answer

gr.close_all()

examples = get_questions(QUESTION_FILE_PATH, QUESTION_LEVEL)

with gr.Blocks() as gaia:
    gr.Markdown("## General AI Assistant - GAIA 🤖🤝🤖")
    gr.Markdown(os.environ.get("DESCRIPTION"))

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                question = gr.Textbox(
                    label="Question *",
                    value=os.environ.get("INPUT_QUESTION", "")
                )
            with gr.Row():
                level = gr.Radio(
                    choices=[1, 2, 3],
                    label="Level",
                    value=int(os.environ.get("INPUT_LEVEL", 1)),
                    scale=1
                )
                ground_truth = gr.Textbox(
                    label="Ground Truth",
                    value=os.environ.get("INPUT_GROUND_TRUTH", ""),
                    scale=1
                )
                file_name = gr.Textbox(
                    label="File Name",
                    scale=2
                )
            with gr.Row():
                openai_api_key = gr.Textbox(
                    label="OpenAI API Key *",
                    type="password"
                )
                gemini_api_key = gr.Textbox(
                    label="Gemini API Key *",
                    type="password"
                )
            with gr.Row():
                clear_btn = gr.ClearButton(
                    components=[question, level, ground_truth, file_name, openai_api_key, gemini_api_key]
                )
                submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            answer = gr.Textbox(
                label="Answer",
                lines=1,
                interactive=False,
                value=os.environ.get("OUTPUT", "")
            )

    clear_btn.click(
        fn=lambda: "",
        outputs=answer
    )
    
    submit_btn.click(
        fn=_run,
        inputs=[question, openai_api_key, gemini_api_key, file_name],
        outputs=answer
    )
    
    gr.Examples(
        examples=examples,
        inputs=[question, level, ground_truth, file_name, openai_api_key, gemini_api_key],
        outputs=answer,
        cache_examples=False
    )

gaia.launch(mcp_server=True)
