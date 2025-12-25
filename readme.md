# IN3SIGHT: Cognitive Forensic Reasoning for Out-of-Context Misinformation Detection

This repository contains the official implementation of **IN3SIGHT**, a cognitive forensic reasoning framework for detecting out-of-context (OOC) image–text misinformation. The framework decomposes the detection process into three principled stages: **Inspection**, **Investigation**, and **Introspection**, without requiring model fine-tuning.

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
│   ├── eval_final.py              # Results after Investigation
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
└── output/                        # Generated intermediate and final results
