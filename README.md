# HermesLog

**HermesLog: A Cloud-Edge Collaborative Fault Diagnosis Framework**

HermesLog is a novel cloud-edge collaborative diagnosis framework. It introduces a medium-sized model as a cognitive relay to enable efficient collaboration between large language models in the cloud and small language models at the edge. The framework leverages cloud large models for deep reasoning on complex faults, while the medium-sized model compresses long reasoning chains into compact knowledge representations. Through a progressive layer-wise alignment mechanism, these representations are then transferred to edge small models. This three-tier architecture ensures diagnostic interpretability while significantly reducing computational overhead and inference latency on edge devices. It enables real-time autonomous diagnosis at the edge, with only uncertain cases escalated to the cloud for deeper analysis. Ultimately, HermesLog maintains high diagnostic accuracy with verifiable explanations while substantially reducing dependency on cloud resources.

## 🔍 Key Features
- **Fault-Oriented Log Filtering and Reasoning (FOLFR)**:
This module extracts diverse fault cases from edge logs via two components. The clustering module groups semantically similar logs using DBSCAN and selects representative samples to form compact sequences, which are then partitioned into cohesive cases based on temporal gaps. For fault identification, the cloud model performs four-stage AutoCoT-Reasoning on suspicious cases to extract evidence-bound clues, followed by reasoning-driven label generation that produces transparent diagnostic results.

- **CRC-Driven Stepwise Layering for Aligned CoT-Reasoning Log Explanation （CRC-DSL）**:
This mechanism enables knowledge transfer from cloud to edge through progressive alignment. The cloud LLM first filters high-confidence samples to build a demonstration set. A medium-sized model then compresses long reasoning chains into compact triplet representations. Edge small models learn through three alignment stages—label, feature, and reasoning chain—using curriculum learning. After alignment, they achieve autonomous diagnosis, performing real-time screening locally with traceable explanations while escalating uncertain cases to the cloud.


## 📁 Key Components
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
  **dataset is divided into three parts, each representing the log data of an client-server, as shown in the following three files:**

- **1.The result of the log sequence after being vectorized by BERT**
```bash
data_{}.npy
```
- **2.Semi-supervised labels, where -1 indicates no label**
```bash
 semi_label_{}.npy
```
- **3.The label of the original data source**
```bash
 label_{}.npy
```

## 📁 Icore code 

1. **Prompt-tuning**   

```bash
claude_zeroshot-cot.py
mistral_fewshot-cot.py
```

2. **Preference-tuning**
```bash
config.py
vllm_sample_offline.py
make_preference.py
run_train.py
```

3. **Knowledge Distillation**
```bash
XXX.py
```

## 📦 Installation

```
conda create --name <env> --file requirements.txt
```




## 📁 Project Structure
```
KDLog/
├── code/               # Icore code (SL-Bert, FL-EMA, docker)
├── data/               # Input logs
├── requirements/       # Create an environment
└── README.md           # Project description
```

```  
 Prompt-tuning/
├── claude_zeroshot-cot-stage1.py
├── data
│   └── output.json
├── mistral_fewshot-cot-stage2.py
├── monitor_gpu.sh
├── output-stage1
│   ├── claude.log
│   └── claude_results.json
├── output-stage2-fewshot-cot
│   ├── mistral.log
│   └── mistral_results.json
├── README.md
├── test_case_id.txt
└── tree.txt
```

```
(base) ➜  dataset ls
claude_zeroshot-cot.py            output-stage2-fewshot-cot
data                              output-stage2-fewshot-nocot
mistral_fewshot-cot-stage2.py     output-stage2-zeroshot-cot
mistral_fewshot-nocot-stage2.py   output-stage2-zeroshot-nocot
mistral_zeroshot-cot-stage2.py    README.md
mistral_zeroshot-nocot-stage2.py  test_case_id.txt
monitor_gpu.sh                    tree.txt
output-stage1
```

```  
Preference-tuning/
├── config.py
├── vllm_sample_offline.py
├── make_preference.py
├── run_train.py
└── vllm_sample_offline.py
```

```
Knowledge Distillation/
``` 

## 🔗 Links
- [Code](https://github.com/IoT1220/KDLog)








## 重要文件


1. 代码

   ```
   cal_acc.py  make_preference.py  openai_multi_client.py  prompts.py         run_train.py  vllm_sample_offline.py
   config.py   make_sft.py         openai_rank.py          run_lora_merge.py  test.py       vllm_sample.py
   ```

2. 中间运行所需配置文件

   ```
   dpo.config.yaml  ipo.config.yaml  lora.merge.yaml  orpo.config.yaml  sft.config.yaml  temp.lora.merge.yaml  temp.yaml
   ```

   以及`data/dataset_info.json`文件

3. 批量运行跑实验的脚本在`scripts`文件夹下

   ```
   sample_aliyun.sh  sample_zte.sh  test_aliyun.sh  test_zte.sh  train_aliyun_sft.sh  train_aliyun.sh  train_zte.sh
   ```

## 运行说明

1. 设置dataset对应的文件夹路径，请编辑`config.py`文件，比如zte对应`ratio1224/ZTE/uncorrected`，请确保该文件夹下有类似于`train1+test1`的文件夹，在`ratio1224/ZTE/uncorrected/train1+test1`文件夹还应该有`train.json`和`test.json`文件

2. 采样模型回复，`vllm_sample_offline.py`

   ```
   CUDA_VISIBLE_DEVICES=0,1,2,3 python vllm_sample_offline.py --model xxxx_path_to_model_folder --dataset zte --fewshot no --sample_n 5 --split train --run_split 1
   ```

   这里面可以指定是否使用in-context learning，`--fewshot no`就是不使用，如果`yes`的话，还可以指定使用哪些fewshot examples：`--fewshot_path xxx_path_to_file`；`--split train`参数指定了使用`train.json`，`--run_split 1`指的是使用`train1+test1`，相似地，如果是`--run_split 2`，那么就是`train2+test2`

3. 模型回复采样结束后，我们需要制作preference data，`make_preference.py`

   - 首先需要把刚才采样得到的文件`xxx.json`放到一个单独的文件夹中，比如`preference_data_folder`

   - 运行命令会得到在同文件夹下的一个`preference_data.train.json`

     ```
     python make_preference.py --dataset zte --input_folder xxx_path_to_preference_data_folder
     ```

   - 这份数据将用来偏好训练模型，我们需要编辑`data/dataset_info.json`，给该数据集命名增加一条：

     ```
         "test_dataset_name": {
             "file_name": "xxxxx_path_to_file_preference_data.train.json",
             "ranking": true,
             "columns": {
                 "prompt": "instruction",
                 "query": "input",
                 "chosen": "chosen",
                 "rejected": "rejected"
             }
         }
     ```

4. 模型偏好训练，`run_train.py`

   ```
   python run_train.py \
           --train_method ipo \
           --dataset test_dataset_name \
           --output output_folder/xxxxxxxx \
           --mode "dpo"
   ```

   其中，`--train_method ipo`可以替换为`dpo`或者`orpo`，注意修改`--output`到你想要的模型保存的位置；训练结束后在output文件夹中你会找到一个`xxx-merged`文件夹，这个就是最终保存的模型ckpt结果

5. 对训练好的模型进行测试，`vllm_sample_offline.py`

   ```
   CUDA_VISIBLE_DEVICES=0,1,2,3 python vllm_sample_offline.py --model xxxx_path_to_merged_model --dataset zte --fewshot yes --sample_n 1 --split test --run_split 1 --fewshot_path xxxx_fewshot.json
   ```

   注意修改`--model`参数为刚才得到的`xxx-merged`文件夹路径，如果使用fewshot，你也可以指定`--fewshot_path xxxx_fewshot.json`参数


   # Knowledge Distillation







