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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_system_message(character):
    return f"Act as {character['name']}. {character['description']}"

@app.post("/api/generate")
async def contextual_adaptation(request: Request):
    data = await request.json()
    character = data['character']
    message = data['message']

    system_message = generate_system_message(character)

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ],
        temperature=0.7,
        max_tokens=500
    )

    return {"response": response.choices[0].message.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
