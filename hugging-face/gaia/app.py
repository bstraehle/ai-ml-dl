import os, threading
import gradio as gr
from crew import run_crew
from util import get_questions

QUESTION_FILE_PATH = "data/gaia_validation.jsonl"
QUESTION_LEVEL     = 1

def invoke(level, question, file_name, ground_truth, openai_api_key, gemini_api_key):
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

demo = gr.Interface(fn=invoke, 
                    inputs=[gr.Radio([1, 2, 3], label="Level", value=int(os.environ["INPUT_LEVEL"])),
                            gr.Textbox(label="Question *", value=os.environ["INPUT_QUESTION"]),
                            gr.Textbox(label="File Name"),
                            gr.Textbox(label="Ground Truth", value=os.environ["INPUT_GROUND_TRUTH"]),
                            gr.Textbox(label="OpenAI API Key *", type="password"),
                            gr.Textbox(label="Gemini API Key *", type="password")],
                    outputs=[gr.Textbox(label="Answer", lines=1, interactive=False, value=os.environ["OUTPUT"])],
                    title="General AI Assistant 🤖🤝🤖",
                    description=os.environ["DESCRIPTION"],
                    examples=get_questions(QUESTION_FILE_PATH, QUESTION_LEVEL),
                    cache_examples=False
                   )

demo.launch()
