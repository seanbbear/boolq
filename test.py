import nlp 
from nlp import load_dataset
from torch.utils.data import DataLoader
import torch
from transformers import AdamW
import os
import wandb
from transformers import AutoTokenizer, AutoModelForMaskedLM

train_dataset = nlp.load_dataset('boolq', split='train')
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

for data in train_dataset:
    data['question'] = tokenizer.__call__(data['question'],return_tensors='pt')
    data['passage'] = tokenizer.__call__(data['passage'],return_tensors='pt')
    if data['answer']==True:
        data['answer'] = tokenizer.__call__('1',return_tensors='pt')
    elif data['answer']==False :
        data['answer'] = tokenizer.__call__('0',return_tensors='pt')

for data in test_dataset:
    data['question'] = tokenizer.__call__(data['question'],return_tensors='pt')
    data['passage'] = tokenizer.__call__(data['passage'],return_tensors='pt')
    if data['answer']==True:
        data['answer'] = tokenizer.__call__('1',return_tensors='pt')
    elif data['answer']==False :
        data['answer'] = tokenizer.__call__('0',return_tensors='pt') 

# 

# print(tokenizer.__call__(['今天天氣很好'],return_tensors='pt'))