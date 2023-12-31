import gradio as gr
import base64, boto3, io, json, os

from PIL import Image

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]

bedrock_runtime = boto3.client(
    aws_access_key_id = aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key,
    service_name = "bedrock-runtime",
    region_name = "us-west-2"
)

def b64_to_pil(img_b64):
    b64_decoded = base64.b64decode(img_b64)
    byte_stream = io.BytesIO(b64_decoded)
    img_pil = Image.open(byte_stream)
    return img_pil

def invoke(prompt, neg_prompt, style_preset):
    if (prompt == ""):
        raise gr.Error("Prompt is required.")
    if (neg_prompt == ""):
        raise gr.Error("Negative prompt is required.")
    if (style_preset == ""):
        raise gr.Error("Style is required.")

    img_pil = None
    
    try:    
        body = json.dumps({"text_prompts": (
                               [{"text": prompt, "weight": 1.0}] +
                               [{"text": neg_prompt, "weight": -1.0}]
                           ),
                           "cfg_scale": 7,
                           "seed": 0,
                           "steps": 150,
                           "style_preset": style_preset.lower().replace(" ", "-"),})
        modelId = "stability.stable-diffusion-xl"
    
        response = bedrock_runtime.invoke_model(body = body, 
                                                modelId = modelId)
    
        response_body = json.loads(response.get('body').read())
        img_b64 = response_body.get('artifacts')[0].get('base64')
        img_pil = b64_to_pil(img_b64)
    except Exception as e:
        completion = e
        
        raise gr.Error(e)
    
    return img_pil

description = """<a href='https://www.gradio.app/'>Gradio</a> UI using the <a href='https://aws.amazon.com/bedrock/'>Amazon Bedrock</a> API 
                 with <a href='https://stability.ai/'>Stability AI</a> Stable Diffusion XL model."""

gr.close_all()

demo = gr.Interface(fn = invoke,
                    inputs = [gr.Textbox(label = "Prompt", lines = 1),
                              gr.Textbox(label = "Negative Prompt", lines = 1),
                              gr.Dropdown(choices = ["Analog Film", "Anime", "Neon Punk", "Photographic", "Pixel Art"], label = "Style"),],
                    outputs = [gr.Image(label="Result")],
                    title = "Generative AI - Text-to-Image",
                    description = description,
                    examples = [["Create a picture of a cat.", "Low quality", "Photographic"],],
                                cache_examples = False)

demo.launch()
