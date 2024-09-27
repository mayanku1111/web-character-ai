import os
import random
import pandas as pd
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from huggingface_hub import login
from huggingface_hub import HfApi
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset

app = Flask(__name__)
CORS(app)

openai.api_key = "API_KEY"
huggingface_token = "HUGGINGFACE_TOKEN"
login(huggingface_token)


def generate_system_message(character):
    prompt = f"Create a system message for a character with the following details:\nName: {character['name']}\nTagline: {character['tagline']}\nDescription: {character['description']}"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that creates system messages for character models."
            },
            {
                "role": "user",
                "content": prompt.strip(),
            }
        ],
        temperature=0.7,
        max_tokens=500,
    )
    return response.choices[0].message['content']


def generate_example(character, system_message, prev_examples, temperature=0.7):
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": f"Generate an example conversation with the character '{character['name']}' based on their description and system message. Format: [Human]: (question) [AI]: (response)"
        }
    ]

    if len(prev_examples) > 0:
        if len(prev_examples) > 5:
            prev_examples = random.sample(prev_examples, 5)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=1354,
    )

    generated_content = response.choices[0].message['content'].strip()
    if not generated_content.startswith('-----------'):
        generated_content = '-----------\n' + generated_content
    if not generated_content.endswith('-----------'):
        generated_content = generated_content + '\n-----------'

    return generated_content


def extract_prompt_response(example):
    parts = example.split('-----------')
    if len(parts) == 4:
        return parts[1].strip(), parts[2].strip()
    else:
        return None, None


@app.route('/api/contextual-adaptation', methods=['POST'])
def contextual_adaptation():
    data = request.json
    character = data['character']
    message = data['message']

    system_message = generate_system_message(character)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ],
        temperature=0.7,
        max_tokens=500
    )

    return jsonify({"response": response.choices[0].message['content']})


@app.route('/api/fine-tune', methods=['POST'])
def fine_tune():
    data = request.json
    character = data['character']
    training_data = data['training_data']

    system_message = generate_system_message(character)
    examples = [generate_example("", [], temperature=0.7) for _ in range(200)]

    prompts = []
    responses = []
    for example in examples:
        prompt, response = extract_prompt_response(example)
        if prompt and response:
            prompts.append(prompt)
            responses.append(response)

    df = pd.DataFrame({
        'prompt': prompts,
        'response': responses
    })

    df = df.drop_duplicates()

    train_df = df.sample(frac=0.9, random_state=42)
    test_df = df.drop(train_df.index)

    train_df.to_json('train.jsonl', orient='records', lines=True)
    test_df.to_json('test.jsonl', orient='records', lines=True)

    model_name = "NousResearch/llama-2-7b-chat-hf"
    new_model = "llama-2-7b-custom"
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    output_dir = "./results"
    num_train_epochs = 1
    fp16 = False
    bf16 = False
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 1
    gradient_checkpointing = True
    max_grad_norm = 0.3
    learning_rate = 2e-4
    weight_decay = 0.001
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "constant"
    max_steps = -1
    warmup_ratio = 0.03
    group_by_length = True
    save_steps = 25
    logging_steps = 5
    max_seq_length = None
    packing = False
    device_map = {"": 0}

    train_dataset = load_dataset('json', data_files='train.jsonl', split="train")
    valid_dataset = load_dataset('json', data_files='test.jsonl', split="train")

    train_dataset_mapped = train_dataset.map(lambda examples: {
        'text': [f'[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n' + prompt + ' [/INST] ' + response for
                 prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)
    valid_dataset_mapped = valid_dataset.map(lambda examples: {
        'text': [f'[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n' + prompt + ' [/INST] ' + response for
                 prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="all",
        evaluation_strategy="steps",
        eval_steps=5
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset_mapped,
        eval_dataset=valid_dataset_mapped,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    trainer.train()
    trainer.model.save_pretrained(new_model)
    HfApi.create_repo(repo_id=f"whitepenguin/{new_model}", private=True, exist_ok=True)
    trainer.model.push_to_hub(f"whitepenguin/{new_model}")
    tokenizer.push_to_hub(f"whitepenguin/{new_model}")

    return jsonify({"message": "Fine-tuning completed successfully"})

from huggingface_hub import HfApi
@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    character = data['character']
    message = data['message']
    use_fine_tuned = data['use_fine_tuned']

    if use_fine_tuned:
        model_name = f"{character['name'].replace(' ', '_').lower()}-character"
        model = AutoModelForCausalLM.from_pretrained(f"whitepenguin/{model_name}")
        tokenizer = AutoTokenizer.from_pretrained(f"whitepenguin/{model_name}")

        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        prompt = f"[INST] {message} [/INST]"
        response = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']

        response = response.split('[/INST]')[-1].strip()
    else:
        system_message = generate_system_message(character)
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=500
        ).choices[0].message['content']

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug = True)
