# HermesLog

**HermesLog: A Cloud-Edge Collaborative Fault Diagnosis Framework**

HermesLog is a novel cloud-edge collaborative diagnosis framework. It introduces a medium-sized language model (MLM, Mistral-7B) as a cognitive relay to enable efficient collaboration between ultra-large language models (LLMs, e.g., GPT-4o and Claude 3.5 sonnet) in the cloud and small language models (SLMs, e.g., Gemma3-4b-it, Phi4mini-3.8b, and Qwen2.5-3b) at the edge. The framework leverages cloud-based LLMs for deep reasoning on complex faults, while the MLM compresses long reasoning chains into compact knowledge representations. Through a progressive layer-wise alignment mechanism, these representations are then transferred to edge-based SLMs. This three-tier architecture ensures diagnostic interpretability while significantly reducing computational overhead and inference latency on edge devices. It enables real-time autonomous diagnosis at the edge, with only uncertain cases escalated to the cloud for deeper analysis. Ultimately, HermesLog maintains high diagnostic accuracy with verifiable explanations while substantially reducing dependency on cloud resources.

## 🔍 Key Features
- **Fault-Oriented Log Filtering and Reasoning (FOLFR)**:
This module extracts diverse fault cases from edge logs via two components. The clustering module groups semantically similar logs using DBSCAN and selects representative samples to form compact sequences, which are then partitioned into cohesive cases based on temporal gaps. For fault identification, the cloud model performs four-stage AutoCoT-Reasoning on suspicious cases to extract evidence-bound clues, followed by reasoning-driven label generation that produces transparent diagnostic results.

- **CRC-Driven Stepwise Layering for Aligned CoT-Reasoning Log Explanation （CRC-DSL）**:
This mechanism enables knowledge transfer from cloud to edge through progressive alignment. The cloud-based LLM first filters high-confidence samples to build a demonstration set. A MLM then compresses long reasoning chains into compact triplet representations. SLMs learn through three alignment stages—label, feature, and reasoning chain—using curriculum learning. After alignment, they achieve autonomous diagnosis, performing real-time screening locally with traceable explanations while escalating uncertain cases to the cloud.


## 😖Key Components
- **Cloud-Edge Collaborationn**: LLM in the cloud for complex reasoning and SLM at the edge for real-time autonomous diagnosis.
- **Four-Stage AutoCoT-Reasoning Process for the Cloud-based LLM**: Four-stage AutoCoT-Reasoning ensures transparent, verifiable fault diagnosis with explicit evidence binding.
- **Stepwise Layered Alignment for the Edge-based SLMs**: MLM bridging enables efficient transfer of complex reasoning capabilities from large to small models.
- **Explainability Edge Autonomy**: SLMs generate compact explanations with traceable evidence, supporting trustworthy decision-making at the edge.
- **Cost Efficiency**: Reduces cloud dependency and inference latency while maintaining high diagnostic accuracy.
  
## 📄 Dataset Description
### This study evaluates two datasets:
  - **The available dataset 1:** link at https://tianchi.aliyun.com/competition/entrance/531947/information.  
  - **The available dataset 2:** link at https://github.com/SycIsDD/LogKG.

### Data storage and load:
  **Our dataset selected a total of 60 cases for cloud-based inference and fine-tuning, and these were randomly and evenly divided into three parts, which were then distributed to three edges.**

## 🪗 Icore code 

1. **Prompt inference and fine-tuning in the cloud**   

```bash
claude_zeroshot-cot.py (LLM)
mistral_pseudo-fewshot-cot.py (MLM)
```

2. **Stepwise alignment training under cloud-edge collaboration**
```bash(cloud)
config.py
vllm_sample_offline.py
make_align.py
run_train.py
```

```bash (edge)
XXX.py
```

## 📦 Installation

```
conda create --name <env> --file requirements.txt
```




## 🍥 Project Structure
```
HermesLog/
├── code/               # Icore code 
├── data/               # Input logs
├── requirements/       # Create an environment
└── README.md           # Project description
```

```  
Prompt-tuning/LLM-MLM
├── Gpt4o_zeroshot-cot_stage1.py
├── data
│   └── output.json
├── mistral_pseudo-fewshot-cot_stage2.py
├── monitor_gpu.sh
├── output_stage1
│   ├── Gpt-4o.log
│   └── Gpt-4o_results.json
├── output_stage2-pseudo-fewshot-cot
│   ├── mistral.log
│   └── mistral_results.json
├── README.md
├── test_case_id.txt
└── tree.txt
```

```
(base) ➜  dataset ls
Gpt4o_zeroshot-cot.py                              output:config-stage1
mistral_pseudo-fewshot-cot-stage2.py               output:vllm_sample_offline-stage2
monitor_gpu.sh                                     tree.txt
output-stage2
```

```  
align-tuning/MLM-SLM
├── config_stage2.py
├── mistral_vllm_sample_offline-tuning_stage2.py
├── make_align-stage3.py                             output:make_align-stage3 
├── run_train_stage3.py                              output:run_train-stage3            
└── vllm_sample_offline.py
```

```  
training/SLM
├── SLM-load-stage4.py
├── SLM_vllm-zero-shot-cot-reason-stage4.py
├── run_trainer_stage4.py  
└── vllm_sample_offline-stage4.py
```  

## 🔗 Links
- [Code]([https://github.com/IoT1220/HermesLog.git])
