#!/bin/python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import random
import argparse
import logging
import re
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.multiwoz.utils import *
from src.multiwoz.utils.config import *
from src.multiwoz.utils.reader import *
from src.multiwoz.postprocess import (
    get_data_split,
    load_schema,
)
from src.utils import *
from chatbots.utils import *
from chatbots.llm import *
from src.multiwoz.schema2function import schema2function
from src.multiwoz.inference import domain2function_mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # arguments for dataset
    parser.add_argument(
        "--dataset_version", type=str, default="2.1", choices=["2.0","2.2", "2.1", "2.3","2.4"]
    )  #
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "dev", "test"]
    )  #
    parser.add_argument(
        "--task", type=str, default="dst", choices=["dst", "e2e", "nlg"]
    )  #
    parser.add_argument(
        "--n_sample", type=int, default=80, help="number of evaluated dialogues"
    )  #
    parser.add_argument(
        "--template",
        type=str,
        default="llama2",
        help="the conversation template of data",
    )  #
    parser.add_argument(
        "--all_turn",
        type=str2bool,
        default=False,
        help="if add all turns of function calls",
    )  #

    args, unknown = parser.parse_known_args()
    print(args)

    # load configuration file and reader (for database query)
    data_prefix = "./data/multiwoz/data/"
    if args.dataset_version == "2.0":
        cfg = Config20(data_prefix)
    elif args.dataset_version == "2.1":
        cfg = Config21(data_prefix)
    elif args.dataset_version == "2.2":
        cfg = Config22(data_prefix)
    elif args.dataset_version == "2.3":
        cfg = Config23(data_prefix)
    elif args.dataset_version == "2.4":
        cfg = Config24(data_prefix)
    reader = MultiWozReader(tokenizer=None, cfg=cfg, data_mode=args.split)

    # load schema, examples, data
    train_data, val_data, test_data = get_data_split(
        dataset_version=args.dataset_version,
        reader=reader,
        n_train=args.n_sample,
        n_val=args.n_sample,
        n_test=args.n_sample,
        return_list=False,
    )
    schema = load_schema(args.dataset_version)

    if args.split == "train":
        data = train_data
    elif args.split == "dev":
        data = val_data
    elif args.split == "test":
        data = test_data

    # save data path
    data_path = f"./data/pre-training_corpora/prompting_data/multiwoz{args.dataset_version}_{args.n_sample}/"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    save_path = f"{data_path}/{args.split}-{args.task}-{args.dataset_version}-v2.jsonl"

    # the conversation template
    conversation = Conversation(
        template_name=args.template,
        function_type="json",
        function_call_prefix=fc_prefix,
        function_call_suffix=fc_suffix,
    )

    # all the processed dials for each task
    processed_data = []

    # prepare input and output for each task
    for dial_id, turns in data.items():
        functions = []
        all_domains = turns[-1]["all_domains"]
        for domain in ["[taxi]", "[train]", "[attraction]", "[hotel]", "[restaurant]"]:
            if domain not in all_domains:
                continue
            for service in schema:
                if service["service_name"] == domain[1:-1]:
                    function = schema2function(
                        service, rename_mapping=domain2function_mapping
                    )
                    functions.append(function)
                    break

        messages = []
        # system instruction
        
        if args.task=="dst":  
            system_messages = [random.choice(dst_instructions)]
            system_messages.extend(tod_notes)
            is_e2e = False
        else:
            system_messages = [random.choice(e2e_instructions)]
            # system_messages.extend(tod_notes)
            is_e2e = True
        system_message = "\n".join(system_messages)
        messages.append({"role": "system", "content": system_message})
        # add conversation messages
        for turn in turns:
            usr = turn["user"]
            resp = turn["nodelx_resp"]
            resp_delx = turn["resp"]
            messages.append({"role": "user", "content": usr})

            action_dict = turn["aspn_dict"] # get current action 
            db = turn["db"] # get number of returned resutls 

            turn_domain = turn["dspn"]
            # ONLY get the last domain in the current turn
            turn_domain = turn_domain.split(" ")[-1]
            # if "] [" in turn_domain:
            #     print(turn_domain)
            turn_bs_dict = turn["turn_bspn_dict"]
            bs_dict = turn["bspn_dict"]

            function_call_dict = {}
            if args.all_turn:  # add function call at all the turns
                if turn_domain in bs_dict:
                    function_call_dict = {
                        "function": domain2function_mapping[turn_domain[1:-1]],
                        "arguments": bs_dict[turn_domain],
                    }
            else:  # only add function call when there are update
                if turn_domain in turn_bs_dict:
                    function_call_dict = {
                        "function": domain2function_mapping[turn_domain[1:-1]],
                        "arguments": bs_dict[turn_domain],
                    }

            if function_call_dict:
                messages.append({"role":"function","content":"<function_call>" + json.dumps(function_call_dict)+"</function_call>"})
                if args.task=="e2e": 
                    messages.append({"role":"observation","content":f"There are {db} entities match"})
                # messages.append(
                #     {
                #         "role": "assistant",
                #         "content": resp,
                #         "function_call": function_call_dict,
                #         "db": db,
                #         "action": action_dict,
                #         "domain": turn_domain
                #     }
                # )
            else:
                if args.task=="e2e": 
                    messages.append({"role":"observation","content":"Do not need to call function"})
            if args.task=="e2e": 
                messages.append({"role":"assistant","content":"Action: " + json.dumps(action_dict) +"\n" + resp_delx})
            else:
                messages.append({"role":"assistant","content": resp})
            
                # messages.append({"role": "assistant", "content": resp, "db": db, "action": action_dict, "domain":turn_domain})
        # if args.task=="e2e":
        #     functions = []
        
        system_prompt = conversation.get_prompt(
            system_message=system_message,
            functions=functions,
            messages=[],
        )
        BEGIN = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'''
        system_prompt = system_prompt.replace(BEGIN,"")
        END = '''<|eot_id|>'''
        system_prompt = system_prompt.replace(END,"")
        # system_prompt = re.sub(BEGIN,"",system_prompt)
        messages[0] = {"role": "system", "content": system_prompt}

        # print(system_prompt)
        # print('='*100)
        
        # construct the prompt, exclude the conversation part
        # system_messages = [random.choice(e2e_instructions)]
        # system_messages.extend(tod_notes)
        # system_message = "\n".join(system_messages)
        # construct each turn of the conversation with function calls
        
        # conversation_prompt = []
        # for message in messages[1:]:
        #     turn_prompt = conversation.get_conversation_ori([message], predict=False, 
        #                                                 is_e2e=is_e2e)
        #     conversation_prompt.append(
        #         {"role": message["role"], "content": turn_prompt}
        #     )
        
        # processed_data.append(
        #     {
        #         "system": system_prompt,
        #         "functions": functions,
        #         "conversation": conversation_prompt,
        #     }
        # )
        
        processed_data.append(
            {
                # "system": system_prompt,
                # "functions": functions,
                "messages": messages,
            }
        )
        
        # construct domain prediction data
        # system_prompt = conversation.get_prompt(
        #     system_message=system_message,
        #     functions=[],
        #     messages=[],
        # )
        # # print(system_prompt)
        # # print('='*100)
        # conversation_prompt = []
        # for message in messages[1:]:
        #     turn_prompt = conversation.get_conversation([message], predict=False, is_multiwoz=True, is_domain_pred=True)
        #     conversation_prompt.append(
        #         {"role": message["role"], "content": turn_prompt}
        # )
        # processed_data.append(
        #     {
        #         "system": system_prompt,
        #         "functions": functions,
        #         "conversation": conversation_prompt,
        #     }
        # )
        # import pdb; pdb.set_trace();


    # summarize
    print(f"Total dialogues: {len(data)}!")
    print(f"Total samples: {len(processed_data)}")

    # save data
    with open(save_path, "w") as file:
        for o in processed_data: 
            file.write(json.dumps(o)+'\n')
            # json.dump(processed_data, file, indent=4)
