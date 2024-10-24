import os
import sys
import math
import torch
import wandb
import logging
import datasets
import argparse
import evaluate
import transformers
import configparser

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.trainer_utils import get_last_checkpoint
from huggingface_hub import login

wandb.init(project='Hanghae99')
wandb.run.name = 'advanced'

config = configparser.ConfigParser()
config.read("secret.ini")
login(config.get("token", "huggingface"))


model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

training_arguments = TrainingArguments(
    output_dir="/tmp/advance_result",
    save_total_limit=1,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_strategy="steps",
    eval_strategy="steps",
    logging_steps=10,
    eval_steps=10,
    num_train_epochs=20
)

def generate_prompt(data):
    prompt_list = []
    for i in range(len(data['instruction'])):
        text = f"### Instruction: {data['instruction'][i]}\n ### Response: {data['output'][i]}"
        prompt_list.append(text)
    return prompt_list

response_template = " ### Response:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


db = Dataset.from_json("4-1\Assignment\Advanced\corpus.json")

print(len(db))

train_split =  db.train_test_split(test_size=0.2)
train_dataset, eval_dataset = train_split['train'], train_split['test']

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=generate_prompt,
    data_collator=collator,
)

checkpoint = None
last_checkpoint = get_last_checkpoint(training_arguments.output_dir)  # 만약 output_dir에 checkpoint가 남아있으면 이를 사용하고, 없으면 None이 return됩니다.
if training_arguments.resume_from_checkpoint is not None:  # output_dir이 아닌 다른 위치에서의 checkpoint를 resume_from_checkpoint로 지정할 수 있습니다.
    checkpoint = training_arguments.resume_from_checkpoint
else:  # 아니면 last_checkpoint로 checkpoint를 지정합니다.
    checkpoint = last_checkpoint

train_result = trainer.train()

trainer.save_model()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()