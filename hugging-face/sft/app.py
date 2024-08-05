# Run full fine-tuning on Google TPU v5e 2x4 or equivalent (220 vCPU, 380 GB RAM, 128 GB VRAM)

import gradio as gr
import os, torch
from datasets import load_dataset
from huggingface_hub import HfApi, login
from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline

ACTION_1 = "Prompt base model"
ACTION_2 = "Fine-tune base model"
ACTION_3 = "Prompt fine-tuned model"

HF_ACCOUNT = "bstraehle"

SYSTEM_PROMPT = "You are a text to SQL query translator. Given a question in English, generate a SQL query based on the provided SQL_CONTEXT. Do not generate any additional text. SQL_CONTEXT: {sql_context}"
USER_PROMPT = "How many new users joined from countries with stricter data privacy laws than the United States in the past month?"
SQL_CONTEXT = "CREATE TABLE users (user_id INT, country VARCHAR(50), joined_date DATE); CREATE TABLE data_privacy_laws (country VARCHAR(50), privacy_level INT); INSERT INTO users (user_id, country, joined_date) VALUES (1, 'USA', '2023-02-15'), (2, 'Germany', '2023-02-27'); INSERT INTO data_privacy_laws (country, privacy_level) VALUES ('USA', 5), ('Germany', 8);"

BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
FT_MODEL_NAME = "Meta-Llama-3.1-8B-text-to-sql"
DATASET_NAME = "gretelai/synthetic_text_to_sql"

def process(action, base_model_name, ft_model_name, dataset_name, system_prompt, user_prompt, sql_context):
    raise gr.Error("Please clone and bring your own Hugging Face credentials.")
    
    if action == ACTION_1:
        result = prompt_model(base_model_name, system_prompt, user_prompt, sql_context)
    elif action == ACTION_2:
        result = fine_tune_model(base_model_name, dataset_name)
    elif action == ACTION_3:
        result = prompt_model(ft_model_name, system_prompt, user_prompt, sql_context)
    return result

def fine_tune_model(base_model_name, dataset_name):
    # Load dataset
    
    dataset = load_dataset(dataset_name)

    print("### Dataset")
    print(dataset)
    print("### Example")
    print(dataset["train"][:1])
    print("###")
    
    # Load model
    
    model, tokenizer = load_model(base_model_name)

    print("### Model")
    print(model)
    print("### Tokenizer")
    print(tokenizer)
    print("###")
        
    # Pre-process dataset
    
    def preprocess(examples):
        model_inputs = tokenizer(examples["sql_prompt"], text_target=examples["sql"], max_length=512, padding="max_length", truncation=True)
        return model_inputs
        
    dataset = dataset.map(preprocess, batched=True)

    print("### Pre-processed dataset")
    print(dataset)
    print("### Example")
    print(dataset["train"][:1])
    print("###")
    
    # Split dataset into training and evaluation sets
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print("### Training dataset")
    print(train_dataset)
    print("### Evaluation dataset")
    print(eval_dataset)
    print("###")
    
    # Configure training arguments

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./{FT_MODEL_NAME}",
        num_train_epochs=3, # 37,500 steps
        #max_steps=1, # overwrites num_train_epochs
        # TODO https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    )

    print("### Training arguments")
    print(training_args)
    print("###")
    
    # Create trainer

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # TODO https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer
    )

    # Train model
    
    trainer.train()

    # Push model and tokenizer to HF

    model.push_to_hub(FT_MODEL_NAME)
    tokenizer.push_to_hub(FT_MODEL_NAME)
    
def prompt_model(model_name, system_prompt, user_prompt, sql_context):
    pipe = pipeline("text-generation", 
                    model=model_name,
                    device_map="auto",
                    max_new_tokens=1000)
    
    messages = [
      {"role": "system", "content": system_prompt.format(sql_context=sql_context)},
      {"role": "user", "content": user_prompt},
      {"role": "assistant", "content": ""}
    ]
    
    output = pipe(messages)
    
    result = output[0]["generated_text"][-1]["content"]

    print("###")
    print(result)
    print("###")
    
    return result

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # TODO: PEFT, LoRA & QLoRA https://huggingface.co/blog/mlabonne/sft-llama3

    return model, tokenizer
    
demo = gr.Interface(fn=process, 
                    inputs=[gr.Radio([ACTION_1, ACTION_2, ACTION_3], label = "Action", value = ACTION_3),
                            gr.Textbox(label = "Base Model Name", value = BASE_MODEL_NAME, lines = 1),
                            gr.Textbox(label = "Fine-Tuned Model Name", value = f"{HF_ACCOUNT}/{FT_MODEL_NAME}", lines = 1),
                            gr.Textbox(label = "Dataset Name", value = DATASET_NAME, lines = 1),
                            gr.Textbox(label = "System Prompt", value = SYSTEM_PROMPT, lines = 2),
                            gr.Textbox(label = "User Prompt", value = USER_PROMPT, lines = 2),
                            gr.Textbox(label = "SQL Context", value = SQL_CONTEXT, lines = 4)],
                    outputs=[gr.Textbox(label = "Prompt Completion", value = os.environ["OUTPUT"])],
                    title = "Supervised Fine-Tuning (SFT)",
                    description = os.environ["DESCRIPTION"])
demo.launch()
