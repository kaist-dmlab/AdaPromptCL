
# AdaPromptCL

One Size Fits All for Semantic Shifts: Adaptive Prompt Tuning for Continual Learning
- This work has been published in the proceedings of ICML 2024.
- [Paper link](https://openreview.net/pdf?id=WUi1AqhKn5)



## About
- The source code of the paper **One Size Fits All for Semantic Shifts: Adaptive Prompt Tuning for Continual Learning**.
- The source code is mainly based on the pytorch implementation of a widely-used environment for [DualPrompt & L2P](https://github.com/JH-LEE-KR/dualprompt-pytorch)
- For semantic-based task grouping of AdaPromptCL, please refer to the function: `uni_or_specific_prompts` in `engine.py`

  

## Environment

The system I used and tested in

- Python 3.8
- Ubuntu 20.04.4 LTS
- pytorch==1.12.1
- torchvision==0.13.1
- Two NVIDIA GeForce RTX 3080
- timm==0.8.0

  

## Usage

Please follow the instructions to train AdaPromptCL:

### Data preparation
- CIFAR100 and ImageNet-R
```
datasets.CIFAR100(download=True)
Imagenet_R(download=True)
```
- VTAB
To download the dataset, refer to [VTAB-1k](https://github.com/google-research/task_adaptation)
For the generation of **VTAB-RecR** and **VTAB-SimS**, please refer to `datasets.py`
  

## Training & Evaluation

The following command lines train and evaluate **AdaPromptCL** on various CL scenarios:

Multi gpus (two NVIDIA GeForce RTX 3080) can be used as the below:

```
# ImageNet-9: Mild shifting scenario     
CUDA_VISIBLE_DEVICES=$gpu_id,$gpu_id2 python -m torch.distributed.launch   \
          --nproc_per_node=2 --master_port=$port_id  --use_env main.py    \
          imr_dualprompt         \
          --model vit_base_patch16_224     \
          --batch-size 64         --data-path "your_path"   \
          --output_dir "your_path" \
          --use_e_prompt --e_prompt_layer_idx 0 1 2 3 4 \
          --seed $seed \
          --data_driven_evolve --uni_or_specific --converge_thrh 0.4  \
          --mergable_prompts --postmerge_thrh 0.6 

# Cifar: Mild shifting scenario     
CUDA_VISIBLE_DEVICES=$gpu_id,$gpu_id2 python -m torch.distributed.launch   \
          --nproc_per_node=2 --master_port=$port_id  --use_env main.py    \
          cifar100_dualprompt         \
          --model vit_base_patch16_224     \
          --batch-size 64         --data-path "your_path"   \
          --output_dir "your_path" \
          --use_e_prompt --e_prompt_layer_idx 0 1 2 3 4 \
          --seed $seed \
          --data_driven_evolve --uni_or_specific --converge_thrh 0.4  \
          --mergable_prompts --postmerge_thrh 0.6 

# VTAB-19T: Abrupt shifting scenario
CUDA_VISIBLE_DEVICES=$gpu_id,$gpu_id2 python -m torch.distributed.launch   \
          --nproc_per_node=2 --master_port=$port_id  --use_env main.py    \
          vtab_dualprompt         \
          --model vit_base_patch16_224     \
          --batch-size 64         --data-path  "your_vtab_path"    \
          --output_dir  "your_path" \
          --use_e_prompt --e_prompt_layer_idx 0 1 2 3 4 \
          --small_dataset \
          --seed $seed \
          --data_driven_evolve  --uni_or_specific --converge_thrh 0.4  \
          --mergable_prompts --postmerge_thrh 0.6 
          
    
# VTAB-Rec10: Varying shifting scenario
# if you change args "num_overlaps", you can change the number of overlaps in tasks
# e.g., if num_overlaps=10, then the dataset becomes VTAB-Rec50 
CUDA_VISIBLE_DEVICES=$gpu_id,$gpu_id2 python -m torch.distributed.launch   \
          --nproc_per_node=2 --master_port=$port_id  --use_env main.py    \
          overlap_vtab_dualprompt         \
          --model vit_base_patch16_224     \
          --batch-size 64         --data-path "your_vtab_path"   \
          --output_dir "your_path" \
          --use_e_prompt --e_prompt_layer_idx 0 1 2 3 4 \
          --small_dataset \
          --vtab_datasets clevr_count resisc45 diabetic_retinopathy oxford_flowers102 caltech101 \
          --num_tasks 5 --seed $seed \
          --data_driven_evolve --uni_or_specific --converge_thrh 0.4 \
          --num_overlaps 2 --num_overlapping_tasks 5 \
          --mergable_prompts --postmerge_thrh 0.6 

# VTAB-Sim25: Varying shifting scenario
# if you change args "overlap_similarity", you can change the semantic similarity between tasks
# e.g., if overlap_similarity=75, then the dataset becomes VTAB-Sim75 
CUDA_VISIBLE_DEVICES=$gpu_id python -m torch.distributed.launch   \
          --nproc_per_node=2 --master_port=$port_id  --use_env main.py    \
          overlap_vtab_dualprompt         \
          --model vit_base_patch16_224     \
          --batch-size 64         --data-path "your_vtab_path"    \
          --output_dir "your_path" \
          --use_e_prompt --e_prompt_layer_idx 0 1 2 3 4 \
          --num_tasks 19 --seed $seed \
          --data_driven_evolve  --uni_or_specific --converge_thrh 0.4  \
          --num_overlaps 1 --num_overlapping_tasks 19 --milder_vtab \
          --overlap_similarity 25 \
          --mergable_prompts --postmerge_thrh 0.6 --no_mild_tasks --num_no_mild_tasks 9 \
          --online_cls_mask # because pre-set class-mask cannot be applied in VTAB-SimS
         

```
# License
This repository is released under the Apache 2.0 license.


  
