import os
import random
import pandas as pd
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from huggingface_hub import login
from huggingface_hub import HfApi

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import json 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

CONVO_FILE = "conversation_history.json"

def load_conversation_history():
    if not os.path.exists(CONVO_FILE):
        with open(CONVO_FILE, "w") as file:
            json.dump([], file) 
    with open(CONVO_FILE, "r") as file:
        return json.load(file)
    return []

def save_conversation_history(conversation):
    with open(CONVO_FILE, "w") as file:
        json.dump(conversation, file)

def generate_system_message(character):
    return f"Act as {character['name']} with {character['tagline']} and description as {character['description']}."

def generate_follow_up_question(message):
    prompt = f"Generate a follow-up question based on the following message: {message}"
    messages = [
        {"role": "system", "content": "You are a character language model"},
        {"role": "user", "content": prompt}
    ]    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=20
    )
    return response.choices[0].message.content


@app.post("/api/generate")
async def contextual_adaptation(request: Request):
    data = await request.json()
    character = data['character']
    message = data['message']

    conversation_history = load_conversation_history()

    system_message = generate_system_message(character)

    messages = [{"role": "system", "content": system_message}]
    
    for entry in conversation_history:
        messages.append({"role": "user", "content": entry["user_message"]})
        messages.append({"role": "assistant", "content": entry["assistant_response"]})

    messages.append({"role": "user", "content": message})

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    assistant_response = response.choices[0].message.content

    new_entry = {
        "user_message": message,
        "assistant_response": assistant_response
    }
    conversation_history.append(new_entry)
    save_conversation_history(conversation_history)

    follow_up_question = generate_follow_up_question(message)

    return {"response": assistant_response, "follow_up_question": follow_up_question}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)