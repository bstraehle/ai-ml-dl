import gradio as gr
import os, torch
from datasets import load_dataset
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Fine-tune on NVidia A10G Large (sleep after 1 hour)

hf_profile = "bstraehle"

action_1 = "Fine-tune pre-trained model"
action_2 = "Prompt fine-tuned model"

system_prompt = "You are a text to SQL query translator. Given a question in English, generate a SQL query based on the provided SCHEMA. Do not generate any additional text. SCHEMA: {schema}"
user_prompt = "What is the total trade value and average price for each trader and stock in the trade_history table?"
schema = "CREATE TABLE trade_history (id INT, trader_id INT, stock VARCHAR(255), price DECIMAL(5,2), quantity INT, trade_time TIMESTAMP);"

base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
dataset = "gretelai/synthetic_text_to_sql"

def prompt_model(model_id, system_prompt, user_prompt, schema):
    pipe = pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    messages = [
      {"role": "system", "content": system_prompt.format(schema=schema)},
      {"role": "user", "content": user_prompt},
      {"role": "assistant", "content": ""}
    ]
    output = pipe(messages)
    result = output[0]["generated_text"][-1]["content"]
    print(result)
    return result

def fine_tune_model(base_model_id):
    tokenizer = download_model(base_model_id)
    fine_tuned_model_id = upload_model(base_model_id, tokenizer)
    return fine_tuned_model_id
        
def download_model(base_model_id):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    model.save_pretrained(base_model_id)
    return tokenizer

#def download_dataset(dataset):
#    ds = load_dataset(dataset)
#    return ""

def upload_model(base_model_id, tokenizer):
    fine_tuned_model_id = replace_hf_profile(base_model_id)
    login(token=os.environ["HF_TOKEN"])
    api = HfApi()
    api.create_repo(repo_id=fine_tuned_model_id)
    api.upload_folder(
        folder_path=base_model_id,
        repo_id=fine_tuned_model_id
    )
    tokenizer.push_to_hub(fine_tuned_model_id)
    return fine_tuned_model_id

def replace_hf_profile(base_model_id):
    model_id = base_model_id[base_model_id.rfind('/')+1:]
    return f"{hf_profile}/{model_id}"

def process(action, base_model_id, dataset, system_prompt, user_prompt, schema):
    if action == action_1:
        result = fine_tune_model(base_model_id)
    elif action == action_2:
        fine_tuned_model_id = replace_hf_profile(base_model_id)
        result = prompt_model(fine_tuned_model_id, system_prompt, user_prompt, schema)
    return result

demo = gr.Interface(fn=process, 
                    inputs=[gr.Radio([action_1, action_2], label = "Action", value = action_1),
                            gr.Textbox(label = "Base Model ID", value = base_model_id, lines = 1),
                            gr.Textbox(label = "Dataset", value = dataset, lines = 1),
                            gr.Textbox(label = "System Prompt", value = system_prompt, lines = 2),
                            gr.Textbox(label = "User Prompt", value = user_prompt, lines = 2),
                            gr.Textbox(label = "Schema", value = schema, lines = 2)],
                    outputs=[gr.Textbox(label = "Completion")])
demo.launch()
