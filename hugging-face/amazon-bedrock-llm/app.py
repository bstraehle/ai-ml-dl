import gradio as gr
import boto3, json, os, wandb

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
wandb_api_key = os.environ["WANDB_API_KEY"]

config = {
    "max_tokens_to_sample": 300,
    "model": "anthropic.claude-v2",
    "temperature": 0,
    "top_k": 250,
    "top_p": 0.999,
}

wandb.login(key = wandb_api_key)
wandb.init(project = "bedrock-llm", config = config)
config = wandb.config

bedrock_runtime = boto3.client(
    aws_access_key_id = aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key,
    service_name = "bedrock-runtime",
    region_name = "us-west-2"
)

def invoke(prompt):
    body = json.dumps({"prompt": "\n\nHuman: " + prompt + "\n\nAssistant: ",
                       "max_tokens_to_sample": config.max_tokens_to_sample,
                       "temperature": config.temperature,
                       "top_k": config.top_k,
                       "top_p": config.top_p,
                       "stop_sequences": ["\n\nHuman: "]
                      })
    modelId = config.model
    accept = "application/json"
    contentType = "application/json"
    #response = bedrock_runtime.invoke_model(body = body, modelId = modelId, accept = accept, contentType = contentType)
    #response_body = json.loads(response.get("body").read())
    #completion = response_body["completion"]
    #wandb.log({"prompt": prompt, "completion": completion})
    #return completion
    return "🛑 Execution is commented out, to view the source code see https://huggingface.co/spaces/bstraehle/amazon-bedrock-llm/tree/main."

description = """<a href='https://www.gradio.app/'>Gradio</a> UI using <a href='https://aws.amazon.com/bedrock/'>Amazon Bedrock</a> API 
                 with <a href='https://www.anthropic.com/'>Anthropic</a> Claude 2 foundation model. 
                 Model performance evaluation via <a href='https://wandb.ai/bstraehle'>Weights & Biases</a>."""

gr.close_all()
demo = gr.Interface(fn=invoke, 
                    inputs = [gr.Textbox(label = "Prompt", lines = 1)],
                    outputs = [gr.Textbox(label = "Completion", lines = 1)],
                    title = "Generative AI - LLM",
                    description = description)
demo.launch()
