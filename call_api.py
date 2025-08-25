import json
import requests
import os

url = "http://10.254.138.189:8030/generate"

prompt = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable assistant trained to provide accurate and helpful information. Please respond to the user's queries promptly and politely.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Hãy viết bài văn tả mẹ làm nghề giáo viên. Hãy trả lời bằng tiếng Việt<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
# prompt = '''
# Viết bài văn tả mẹ? Hãy trả lời bằng tiếng Việt
# '''
data_tgi = {
    "inputs": prompt,
     "parameters": {
        "temperature": 0.5,
        "top_p": 0.5,
        "max_new_tokens": 256,
        # "n_seqs": 1,
        "stop":["Example","assistant"],
         "best_of": 1,
         "do_sample": False,
         "repetition_penalty": 1.03,
     }
}
response = requests.post(url, json=data_tgi)
print(response)
print(response.json())
outputs = response.json()["generated_text"]
# print(outputs)