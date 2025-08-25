# Spec-TOD: A specialized Instruction-Tuned LLM Framework for Eficient Task-Oriented Dialogue System [Link paper](https://arxiv.org/abs/2507.04841)
## Data Prepreration
We construct data for fintuning and evaluation base on previous works: [FncTOD](https://github.com/facebookresearch/FnCTOD/tree/main), [UBAR](https://github.com/TonyNemo/UBAR-MultiWOZ). Refer to [FnCTOD](https://github.com/facebookresearch/FnCTOD/tree/main) for more details. 
### Collect data from different datasets.
```
cd sh_folders
sh processing-sgd.sh
sh prompting-sgd.sh
... more dataset
sh create_finetunedata.sh
```
### Convert multiwoz dataset to instruction finetuning format
To create data for finetuning, run script 
```
cd sh_folders
sh prompting-multiwoz.sh
```
You can see our processed data in `data/share_gpt_format`
## Finetuning
## Environment setup
We use axolotl framework to finetune our model. 

```
docker pull winglian/axolotl:main-20240807-py3.10-cu121-2.3.1
docker run --dit --name axolotl -v /home:/home --gpus all winglian/axolotl:main-20240807-py3.10-cu121-2.3.1
cd finetune_axolotl
CUDA_VISIBLE_DEVICES="0,1" accerlerate launch --main_process_port 29498 -m axolotl.cli.train fune_tune_10_percent.yml --debug
```

## Inference

### 1. Set up TGI server
First, we need merge lora checkpoint to the original model
```
cd finetune_axolotl
CUDA_VISIBLE_DEVICES="0" python3 -m axolotl.cli.merge_lora merge_10_percent.yml --lora_model_dir="your_dir"
```

To accelerate inference, model is deployed to TGI server (https://github.com/huggingface/text-generation-inference).
```
bash run_tgi_docker.sh
```

### 2. Inference
#### Environment setup
```
pip install -r requirements.txt
```
We divide test set to multiple files to infer parallel.
```
cd sh_folders
python3 script_scirpt_tgi.py
cd infer_e2e_fnctod-llama3-8b-axolotl-tgi-few-shot
sh parent_script.sh
```
Then merge these files into one file to calculate scores.
```
cd outputs/multiwoz2.2
python3 merge.py
cd ../../sh_folders/
sh inference_fnctod-llama-e2e.sh
```
Do similar if you want to benchmark DST task.

## Acknowledgement

This code is adapted and modified upon the released code  [FnCTOD](https://github.com/facebookresearch/FnCTOD/tree/main) at ACL 2024. 

We appreciate their open-sourcing such high-quality code, which is very helpful to our research.



## License

## Citation
@misc{nguyen2025spectodspecializedinstructiontunedllm,
      title={Spec-TOD: A Specialized Instruction-Tuned LLM Framework for Efficient Task-Oriented Dialogue Systems}, 
      author={Quang-Vinh Nguyen and Quang-Chieu Nguyen and Hoang Pham and Khac-Hoai Nam Bui},
      year={2025},
      eprint={2507.04841},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.04841}, 
}
