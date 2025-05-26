import os, threading
import gradio as gr
from crew import run_crew
from utils import get_questions

QUESTION_FILE_PATH = "data/gaia_validation.jsonl"

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

examples_1 = get_questions(QUESTION_FILE_PATH, 1)
examples_2 = get_questions(QUESTION_FILE_PATH, 2)
examples_3 = get_questions(QUESTION_FILE_PATH, 3)

with gr.Blocks() as gaia:
    gr.Markdown("## General AI Assistant 🧠")
    gr.Markdown(os.environ.get("DESCRIPTION"))

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                question = gr.Textbox(
                    label="Question *"
                )
            with gr.Row():
                level = gr.Radio(
                    choices=[1, 2, 3],
                    label="Level",
                    scale=1
                )
                ground_truth = gr.Textbox(
                    label="Ground Truth",
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
                interactive=False
            )

    #clear_btn.click(
    #    fn=lambda: "",
    #    outputs=answer
    #)
    
    submit_btn.click(
        fn=_run,
        inputs=[question, openai_api_key, gemini_api_key, file_name],
        outputs=answer
    )
    
    gr.Examples(
        label="Level 1",
        examples=examples_1,
        inputs=[question, level, ground_truth, file_name, openai_api_key, gemini_api_key],
        outputs=answer,
        cache_examples=False
    )

    gr.Examples(
        label="Level 2",
        examples=examples_2,
        inputs=[question, level, ground_truth, file_name, openai_api_key, gemini_api_key],
        outputs=answer,
        cache_examples=False
    )

    gr.Examples(
        label="Level 3",
        examples=examples_3,
        inputs=[question, level, ground_truth, file_name, openai_api_key, gemini_api_key],
        outputs=answer,
        cache_examples=False
    )

gaia.launch(mcp_server=True)
