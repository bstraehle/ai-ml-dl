import gradio as gr

from openai import OpenAI

config = {
    "max_tokens": 1000,
    "model": "gpt-4",
    "temperature": 0,
}

def invoke(openai_api_key, prompt):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (prompt == ""):
        raise gr.Error("Prompt is required.")

    content = ""
    
    try:
        client = OpenAI(api_key = openai_api_key)
    
        completion = client.chat.completions.create(
            max_tokens = config["max_tokens"],
            messages = [{"role": "user", "content": prompt}],
            model = config["model"],
            temperature = config["temperature"],)
    
        content = completion.choices[0].message.content
    except Exception as e:
        err_msg = e

        raise gr.Error(e)

    return content

description = """<a href='https://www.gradio.app/'>Gradio</a> UI using the <a href='https://openai.com/'>OpenAI</a> API 
                 with <a href='https://openai.com/research/gpt-4'>gpt-4</a> model."""

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1),
                              gr.Textbox(label = "Prompt", lines = 1)],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    title = "Generative AI - LLM",
                    description = description,)

demo.launch()
