import os, json
import pandas as pd
path_to_json = "parallel-fnctod-llama3-8b-axolotl-tgi-few-shot"
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print(len(json_files)) 

all_data = {}
chunk_size = 10
prefix = "-test1000"

file_name = prefix[1:] + json_files[0].split(prefix)[1]
for file in json_files:
    with open(f"{path_to_json}/"+file) as f:
        data = json.load(f)
        index = int(file.split(prefix)[0])
        keys = list(data.keys())[index:index+chunk_size]
        for o in keys:
            all_data[o] = data[o]
        # break


with open(file_name,"w") as f:
    json.dump(all_data,f,indent=4)