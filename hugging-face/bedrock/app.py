import gradio as gr
import boto3, json, os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]

config = {
    "max_tokens": 1000,
    "model": "anthropic.claude-3-opus-20240229-v1:0",
    "temperature": 0,
    "top_k": 250,
    "top_p": 0.999,
}

bedrock_runtime = boto3.client(
    aws_access_key_id = aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key,
    service_name = "bedrock-runtime",
    region_name = "us-west-2"
)

def invoke(prompt):
    if not prompt:
        raise gr.Error("Prompt is required.")

    raise gr.Error("Please clone and bring your own credentials.")
    
    completion = ""
    
    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            "system": "You are a honest, helpful, and harmless bot."
        }
        model_id = config["model"]
        model_kwargs =  { 
            "max_tokens": config["max_tokens"],
            "stop_sequences": ["\n\nHuman"],
            "temperature": config["temperature"],
            "top_k": config["top_k"],
            "top_p": config["top_p"]
        }
        body.update(model_kwargs)
        
        response = bedrock_runtime.invoke_model(modelId=model_id,
                                                body=json.dumps(body))
        
        response_body = json.loads(response.get("body").read())
        completion = response_body.get("content", [])[0].get("text", "")
    except Exception as e:
        completion = e
        
        raise gr.Error(e)
    
    return completion

description = """<a href='https://www.gradio.app/'>Gradio</a> UI using the <a href='https://aws.amazon.com/bedrock/'>Amazon Bedrock</a> SDK 
                 with <a href='https://www.anthropic.com/'>Anthropic</a> Claude 3 model."""

gr.close_all()

demo = gr.Interface(fn = invoke, 
                    inputs = [gr.Textbox(label = "Prompt", value = "If I dry one shirt in the sun, it takes 1 hour. How long do 3 shirts take?", lines = 1)],
                    outputs = [gr.Textbox(label = "Completion", lines = 1, value = os.environ["COMPLETION"])],
                    description = description)

demo.launch()
