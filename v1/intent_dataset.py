import os
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict

from src.os_intent.v1.tokenizer import Tokenizer
from src.os_intent.v1.data_loader import create_dataloaders


dataset_name = 'intent'
def get_dataset_loader(tokenizer: Tokenizer, batch_size, test_dataset_path, train_dataset_path_list, use_cache=True, multi_gpus=True):
    cache_dict = {
        'train': f'./cache/{dataset_name}_dataset_cache_train.cache',
        'test': f'./cache/{dataset_name}_dataset_cache_test.cache',
        'validation': f'./cache/{dataset_name}_dataset_cache_validation.cache'
    }

    if not use_cache:
        for path in cache_dict.values():
            if os.path.exists(path):
                os.remove(path)
    dataset_list = []
    
    for dataset_path in train_dataset_path_list:
        dataset = load_dataset('json', data_files=dataset_path)
        unused_columns = set(dataset['train'].features.keys()) - {"text", "label_text"}
        dataset = dataset.remove_columns(unused_columns)
        dataset_list.append(dataset['train'])

    train_dataset = concatenate_datasets(dataset_list)
    test_dataset = load_dataset('json', data_files={"test": test_dataset_path})

    raw_datasets = DatasetDict({"train": train_dataset, "test": test_dataset["test"]})

    tokenizer.input_column = "text"
    tokenizer.output_column = "label_text"

    tokenized_datasets = raw_datasets.map(tokenizer.tokenize_function, batched=True, cache_file_names=cache_dict, batch_size=1)
    tokenized_datasets.set_format("torch")

    train_dataloader, eval_dataloader = create_dataloaders(tokenized_datasets['train'], tokenized_datasets['test'], train_batch_size=batch_size, eval_batch_size=batch_size, multi_gpus=multi_gpus)

    return train_dataloader, eval_dataloader