<div align="center">
  <img src="In3sight.png" width="300"/>

##  <img src="logo.png" width="50" height="50"> IN³SIGHT: Toward Cognitive Forensic Reasoning for OOC Misinformation Detection


This repository contains the official implementation of **IN3SIGHT**, a cognitive forensic reasoning framework for detecting out-of-context (OOC) image–text misinformation. The framework decomposes the detection process into three principled stages: **Inspection**, **Investigation**, and **Introspection**, without requiring model fine-tuning.

Kaige Li, Xiaochun Cao*, IEEE Senior Member

*Corresponding author: [Xiaochun Cao](https://scholar.google.com/citations?user=PDgp6OkAAAAJ&hl=en).

---

## 1. Environment Setup

We recommend using **Python 3.10** with Conda.

```bash
conda create -n IN3SIGHT python=3.10 -y
conda activate IN3SIGHT

pip install -r requirements.txt
pip install flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

All experiments reported in the paper were conducted on a single NVIDIA A100 GPU.


## 2. Repository Structure

```text
.
├── inspection_reasoning.py        # Inspection stage: intrinsic judgment
├── investigation_reasoning.py     # Investigation stage: external evidence audit
├── introspection_reasoning.py     # Introspection stage: evidence-aware reconciliation
│
├── eval/                           # Evaluation utilities
│   ├── eval.py                    # Results after Inspection
│   ├── eval_middle.py              # Results after Investigation
│   └── eval_joint.py              # Results after Introspection
│
├── datasets/                      # Datasets and evidence resources
│   ├── output_*                   # NewsClippings formatted datasets
│   ├── test_external_info_llama_final.json
│   └── *.py                       # Evidence retrieval / translation scripts
│
├── scripts/                       # Optimization and utility scripts
│   ├── shallow_optimizer.py       # Prompt optimization for Inspection
│   ├── Dataset.py                 # Dataset construction
│   └── *.py                       # Evidence filtering & relevance verification
│
├── utils/ # Unified LLM / MLLM utility modules
│   ├── init.py
│   ├── llama_utils.py # Llama-based LLM inference utilities
│   ├── llama_utils_batch.py # Batched inference and acceleration for Llama models
│   ├── Qwen_utils.py # Qwen-VL MLLM inference utilities
│   └── Qwen_instruct_util.py # Instruction-format wrappers for Qwen-VL models
│
└── output/                        # Generated intermediate and final results
```


## 3. Method Overview

🔥 Pending



## 4. Datasets

- **[NewsCLIPpings](https://github.com/g-luo/news_clippings)**  
  Used for prompt calibration and primary evaluation.

- **[VERITE](https://github.com/stevejpapad/image-text-verification)**  
  Used for cross-dataset generalization evaluation.

The file:
```bash
datasets/test_external_info_llama_final.json
```
contains pre-retrieved external evidence for the NewsCLIPpings test set.



## Reproducibility Notes

- No model parameters are updated at any stage.
- All performance gains arise from structured cognitive reasoning.
- Prompt optimization is lightweight and robust to initialization.
- The framework is model-agnostic and transferable to other MLLMs.





## Code Availability Statement
This code is associated with a paper currently under review. To comply with the review process, the code will be made FULLY available once the paper is accepted.  :smiley:

We appreciate your understanding and patience. Once the code is released, we will warmly welcome any feedback and suggestions. Please stay tuned for our updates!











