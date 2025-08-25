import argparse
import json
import os
from typing import Iterable, List

import requests
import sys
import re

from transformers import AutoTokenizer

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

from huggingface_hub import InferenceClient


def get_tokenizer(tgi_ip: str) -> str:
    info = requests.get(f"{tgi_ip}/info").json()
    model_id = "/workspace/home/NLP_CORE/HUB_LLM/" + info["model_id"].split("/")[-1]
    print(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer


tgi_ip = "http://10.254.138.189:8030"
# tokenizer = get_tokenizer(tgi_ip)
tokenizer = AutoTokenizer.from_pretrained("/home/LFew-TOD/ckpt/lora-out-axolotl/Llama-3-8B-Instruct-64-32-2024-06-30-1percent-inputoutput/checkpoint-2988/merged")
tokenizer.eos_token = "<|eot_id|>"
client = InferenceClient(tgi_ip)


def get_result(prompt, history):
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a knowledgeable assistant trained to provide accurate and helpful information. Please respond to the user's queries promptly and politely."}] + history + [{"role": "user", "content": prompt}], 
        tokenize=False, 
        add_generation_prompt=True
    )
    print(prompt)
    return client.text_generation(
        prompt, 
        max_new_tokens=256, 
        stream=True, 
#         repetition_penalty=1.1, 
        do_sample=True,
        stop_sequences=[tokenizer.eos_token]
    )
