import json

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence

with open("data/amazon_massive_scenario_zh-CN/label_to_id.json", "r", encoding="utf-8") as f:
    label_dict_label_to_id = {list(item.keys())[0] :list(item.values())[0] for item in json.load(f)}

with open("data/amazon_massive_scenario_zh-CN/id_to_label.json", "r", encoding="utf-8") as f:
    label_dict_id_to_label = {list(item.keys())[0] :list(item.values())[0] for item in json.load(f)}

def dynamic_padding_collate_fn(batch):
    """动态padding，每个batch内统一到最长序列长度"""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    text = [item["text"] for item in batch]

    
    # 对input_ids和attention_mask进行padding，pad值为0
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    # 对labels进行padding，pad值为-100
    labels = pad_sequence(torch.tensor([label_dict_label_to_id[label] for label in labels]).unsqueeze(0), batch_first=True, padding_value=-100).squeeze()
    attention_mask = pad_sequence(attention_mask,  batch_first=True, padding_value=-100)
    return {
        'text': text,
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }


def create_dataloaders(train_dataset, eval_dataset, train_batch_size=8, eval_batch_size=32, multi_gpus=True):
    if multi_gpus:
        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, collate_fn=dynamic_padding_collate_fn, sampler=DistributedSampler(train_dataset)
        )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=eval_batch_size, collate_fn=dynamic_padding_collate_fn, sampler=DistributedSampler(eval_dataset)
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, collate_fn=dynamic_padding_collate_fn, sampler=RandomSampler(train_dataset)
        )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=eval_batch_size, collate_fn=dynamic_padding_collate_fn, sampler=RandomSampler(eval_dataset)
        )


    return train_dataloader, eval_dataloader