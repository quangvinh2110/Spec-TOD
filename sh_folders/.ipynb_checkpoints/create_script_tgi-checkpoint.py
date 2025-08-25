import os 

model_name = "fnctod-llama3-8b-axolotl-tgi-few-shot"
task = "e2e"
new_dir = f"infer_{task}_{model_name}"
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

n_eval = 1000
begin_index = []
end_index = []
number_of_chunk = 100
chunk_size = int(n_eval / number_of_chunk)

for i in range(0,n_eval,chunk_size):
    begin_index.append(i)
    end_index.append(i+chunk_size)
print(begin_index)
print(end_index)

for i in range(len(begin_index)):
    script = f'''
cd ../../
for dataset_version in 2.0
do
    for split in test
    do
        for n_eval in {n_eval}  
	do
            for multi_domain in False
            do
                for ref_domain in False
                do
                    for ref_bs in False
                    do
                        for add_prev in True
                        do
                            for task in {task} #e2e
                            do
                                for dst_nshot in 0
                                do
                                    for nlg_nshot in 0
                                    do
                                        for function_type in json # text
                                        do
                                            for model in {model_name}
                                            do
                                                python -m src.multiwoz.inference_{task} \
                                                        --dataset_version $dataset_version \
                                                        --target_domains $target_domains \
                                                        --split $split \
                                                        --n_eval $n_eval \
                                                        --model $model \
                                                        --task $task \
                                                        --dst_nshot $dst_nshot \
                                                        --nlg_nshot $nlg_nshot \
                                                        --add_prev $add_prev \
                                                        --ref_domain $ref_domain \
                                                        --ref_bs $ref_bs \
                                                        --multi_domain $multi_domain \
                                                        --function_type $function_type \
                                                        --generate \
                                                        --infer_with_tgi \
                                                        --begin_index {begin_index[i]} \
                                                        --end_index {end_index[i]} \
                                                        --parallel_infer
                                                        # --verbose \
                                                        # --debug
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

'''

    with open(new_dir+f"/script_{i}.sh","w") as f:
        f.write(script)
# print(script)
parent_script = ""
for i in range(len(begin_index)):
    parent_script += f"sh script_{i}.sh &\n"

with open(new_dir+f"/parent_script.sh","w") as f:
        f.write(parent_script)
