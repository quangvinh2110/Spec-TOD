#!/bin/python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import argparse
from typing import List

from chatbots.llm import *
from chatbots.configs import llm_configs

parser = argparse.ArgumentParser(description="Initialize the LLM model.")
parser.add_argument(
    "--model", type=str, default="vicuna-13b", help="Model name for LLM."
)

args = parser.parse_args()

app = FastAPI()


class Prompt(BaseModel):
    input: str
    temperature: float = 0.5
    top_p: float = 1.0
    max_tokens: int = 64
    n_seqs: int = 1
    stop: List[str] = ["\n", "\n\n", "User", "Example"]
    

if args.model in llm_configs:
    model_name = llm_configs[args.model]["model_name"]
    port = llm_configs[args.model]["port"]
else:
    raise ValueError("the model type is not supported")

llm = LLM(model_name=model_name)

# local test
print(
    llm.generate(
        prompt="<|start_header_id|>system<|end_header_id|>\n\nYou are a task-oriented assistant. Your role is to determine which domain the user is seeking information about or attempting to make a booking in during each turn of the conversation. Select the most relevant domain from the following options: [restaurant], [hotel], [taxi], [train], [hospital], [police], [attraction]. If the user's inquiry does not align with a specific domain, use: [general]. Note that the [attraction] domain encompasses various categories, including architecture, boat, cinema, college, concert hall, entertainment, museum, sports activities, nightclub, park, swimming pool, and theatre.\n\n<EXAMPLE>\n<|start_header_id|>user<|end_header_id|>\nhi, could i find some museum in the center of the town ?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n<domain>[attraction]</domain>The railroad museum would be nice for you .<|eot_id|><|start_header_id|>user<|end_header_id|>\ngreat , and i also want to book a taxi to leave the attraction by 08:00 . get contact number and car type .<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n<domain>[taxi]</domain>\n</EXAMPLE>\n\n<EXAMPLE>\n<|start_header_id|>user<|end_header_id|>\nplease find me a place to dine that serves vegetarian food .<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n<domain>[restaurant]</domain>i found a cheap one that serves korea food .\n</EXAMPLE>\n\n<EXAMPLE>\n<|start_header_id|>user<|end_header_id|>\ni am also looking for place -s to go in town . i would love for it to be sports related .<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n<domain>[attraction]</domain>we have 4 swimming pool location -s . what do you think about those ?<|eot_id|><|start_header_id|>user<|end_header_id|>\nokay, thank you . have a good day !<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n<domain>[general]</domain>you too, bye !\n</EXAMPLE>\n\n<EXAMPLE>\n<|start_header_id|>user<|end_header_id|>\ndo you have any place -s to stay in the west that include free parking ?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n<domain>[hotel]</domain>yes, what price range are you looking for ?\n</EXAMPLE>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\ni am trying to find an expensive restaurant in the centre part of town .<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n<domain>[",
        temperature=0.5,
        top_p=1.0,
        max_tokens=32,
        n_seqs=1,
        functions = [],
        function_call = {},
        stop=["Example", "\n", "\n\n"],
    )
)


@app.post("/generate/")
async def generate_text(prompt: Prompt):
    try:
        generations = llm.generate(
            prompt=prompt.input,
            temperature=prompt.temperature,
            top_p=prompt.top_p,
            max_tokens=prompt.max_tokens,
            n_seqs=prompt.n_seqs,
            stop=prompt.stop,
            functions = [],
            function_call = {},
        )
        return {"generated_text": generations}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)
