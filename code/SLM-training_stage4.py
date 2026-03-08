from load_data import JsonToDataFrame
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
import transformers
from trl import SFTTrainer  
from Trainer import CustomTrainer_noKL_noSelfLoss
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer
)
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
from datasets import Dataset 
import torch
import time
import argparse
import logging
import os
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="edgeLogLM SLM distill")
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--dataset", type=str, default="aliyun")
parser.add_argument("--part", type=int, default=1, help="Part")
parser.add_argument("--llm", type=str, default="16+4", help="LLM")
parser.add_argument("--slm", type=int, default=3, help="SLM type")
parser.add_argument("--base_path", type=str, required=True, help="Path to the base folder")
parser.add_argument("--epoch", type=int, default=20, help="Part")
parser.add_argument("--lr", type=float, default=0.001, help="Part")
parser.add_argument("--lora_r", type=int, default=8, help="lora_r")
parser.add_argument("--lora_alpha", type=int, default=16, help="lora_alpha")
parser.add_argument("--lora_dropout", type=float, default=0.2, help="lora_dropout")
parser.add_argument("--max_grad_norm", type=float, default=0.3, help="max_grad_norm")
parser.add_argument("--output_dir", type=str, default="0312")


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda:{}'.format(args.gpu_id))
dataset = args.dataset
part_num = args.part
base_path = args.base_path
epoch = args.epoch
lr = args.lr
llm_type = args.llm
model_type = args.slm
lora_r = args.lora_r
lora_alpha = args.lora_alpha
lora_dropout = args.lora_dropout
output_dir = args.output_dir
max_grad_norm = args.max_grad_norm

if model_type == 1:
    student_model_path = ""
elif model_type == 2:
    student_model_path = ""
elif model_type == 3:
    student_model_path = ""

train_data_path = f"{base_path}/part_{part_num}.json"
save_path = f"{base_path}/Output/{output_dir}_part_{part_num}_{llm_type}_{model_type}slm_{epoch}epoch_{lr}lr_{lora_r}r_{lora_alpha}alpha_{lora_dropout}dropout"

def load_dataset(data_path, tokenizer):
    json_loader = JsonToDataFrame(data_path)
    df = json_loader.load_json_to_dataframe()
    dataset = Dataset.from_pandas(df)
    
    def format_instruction(example):
        assistant_content = example['assistant_content']
        reason_part = assistant_content.split("Label: ")[0].replace("Reason: ", "")
        label = assistant_content.split("Label: ")[1].strip()
        return {
                "messages": [
                    {"role": "user", "content": example['user_content']},
                    {"role": "assistant", "content": f"<reason>Reason: {reason_part}</reason>\n\n<label>Label: {label}</label>"}
                ]
            }
    
    formatted_dataset = dataset.map(format_instruction, batched=False, remove_columns=dataset.column_names)

    def apply_chat_template(examples):
        formatted = [tokenizer.apply_chat_template(dialogue, tokenize=False) 
                     for dialogue in examples["messages"]]
        
        examples["formatted_text"] = formatted
        
        tokenized = tokenizer(
            formatted,
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
            return_offsets_mapping=False
        )
        return tokenized

    formatted_dataset = formatted_dataset.map(apply_chat_template, batched=True)
    formatted_dataset = formatted_dataset.remove_columns(["messages"])
    
    return formatted_dataset


if __name__ == "__main__":
    timestamp = time.strftime('%Y%m%d%H%M%S')
    device="cuda:0"
    student_tokenizer = AutoTokenizer.from_pretrained(
        student_model_path,
        trust_remote_code=True
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_path,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to("cuda")
    
    special_tokens = ["<reason>", "</reason>", "<label>", "</label>"]
    student_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    student_tokenizer.pad_token = student_tokenizer.eos_token 

    data = load_dataset(train_data_path, student_tokenizer)
    total_train_samples = len(data)
    
    print(f"Model is on device: {student_model.device}")

    student_model.resize_token_embeddings(len(student_tokenizer))
    student_model.gradient_checkpointing_enable()
    
    model = prepare_model_for_kbit_training(student_model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj","down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config).to(model.device)

    training_args = transformers.TrainingArguments(
                num_train_epochs=epoch,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                gradient_checkpointing=True,
                save_total_limit=1,
                logging_strategy="steps",
                logging_steps=1,
                learning_rate=lr,
                fp16=True,
                output_dir=f"{save_path}/model_save",
                optim="paged_adamw_32bit",
                max_grad_norm=max_grad_norm,
                warmup_ratio=0.03,
                lr_scheduler_type="cosine",
                remove_unused_columns=False
            )
    data = data.remove_columns(["formatted_text"])
    data_collator=DataCollatorForLanguageModeling(tokenizer=student_tokenizer, mlm=False)
    trainer = CustomTrainer_noKL_noSelfLoss(
        model=model,
        train_dataset=data,
        log_save_path=save_path,
        data_collator=data_collator,
        args=training_args
    )
    
    model.config.use_cache = False
    trainer.train()
    
    trainer.save_model(trainer.args.output_dir)
    student_tokenizer.save_pretrained(trainer.args.output_dir)
    final_model = trainer.model.merge_and_unload()
    final_model.save_pretrained(f"{save_path}/model_save_final")
    student_tokenizer.save_pretrained(f"{save_path}/model_save_final")



