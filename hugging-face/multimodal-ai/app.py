import gradio as gr
import base64, os, threading

from openai import OpenAI

lock = threading.Lock()

config = {
    "max_tokens": 1000,
    "model": "gpt-4o",
    "temperature": 0
}

def get_img_b64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def invoke(openai_api_key, prompt, image):
    if (openai_api_key == ""):
        raise gr.Error("OpenAI API Key is required.")
    if (prompt == ""):
        raise gr.Error("Prompt is required.")
    if (image is None):
        raise gr.Error("Image is required.")

    with lock:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        content = ""
        
        try:
            client = OpenAI()
    
            img_b64 = get_img_b64(image)
    
            completion = client.chat.completions.create(
                max_tokens = config["max_tokens"],
                messages = [{"role": "user",
                             "content": [{"type": "text", 
                                          "text": prompt},
                                         {"type": "image_url",
                                          "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
                model = config["model"],
                temperature = config["temperature"]
            )
        
            content = completion.choices[0].message.content
        except Exception as e:
            err_msg = e
            raise gr.Error(e)
        finally:
            del os.environ["OPENAI_API_KEY"]        
    
        return content

gr.close_all()

demo = gr.Interface(
    fn = invoke, 
    inputs = [gr.Textbox(label = "OpenAI API Key", type = "password", lines = 1),
              gr.Textbox(label = "Prompt", lines = 1, value = "Describe the diagram"),
              gr.Image(label = "Image", type = "filepath", sources = ["upload"], 
                       value = "https://raw.githubusercontent.com/bstraehle/ai-ml-dl/main/hugging-face/multimodal-ai/architecture.png")],
    outputs = [gr.Markdown(label = "Completion", value = os.environ["OUTPUT"])],
    title = "Multimodal Reasoning Application",
    description = os.environ["DESCRIPTION"]
)

demo.launch()
