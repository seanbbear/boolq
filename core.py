import logging
from nlp import load_dataset
import numpy as np
import torch
from torch.utils.data import TensorDataset

def get_dataset(name, tokenizer, split):     
    dataset = load_dataset(name, split=split)

    data_len = len(dataset)      
    
    input_ids = np.zeros(shape=(data_len,512))     
    token_type_ids = np.zeros(shape=(data_len,512))     
    attention_mask = np.zeros(shape=(data_len,512))     
    answer = []          
    
    # count = 0     
    for i in range(len(dataset)):         
        tensor_features = tokenizer.__call__(dataset[i]['question'], dataset[i]['passage'], stride=128, return_tensors='np', max_length = 512,  padding='max_length', truncation=True,return_overflowing_tokens=True)          

        # append越來越慢 https://hant-kb.kutu66.com/others/post_544244         
        input_ids[i] = tensor_features['input_ids']              
        token_type_ids[i] = tensor_features['token_type_ids']         
        attention_mask[i] = tensor_features['attention_mask']

        if dataset[i]['answer']==True:             
            answer.append(1)         
        elif dataset[i]['answer']==False:             
            answer.append(0)

        # if i==10:             
        #     break         
        # count+=1
        

    input_ids = torch.LongTensor(input_ids)     
    token_type_ids = torch.LongTensor(token_type_ids)      
    attention_mask = torch.LongTensor(attention_mask)     
    answer = torch.LongTensor(answer)      
    
    return TensorDataset(input_ids, token_type_ids, attention_mask, answer)

def compute_accuracy(y_pred, y_target):
    # 計算正確率
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100  



    


































