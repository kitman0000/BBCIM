import torch

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.input_column = None
        self.output_column = None

    def tokenize_function(self, examples):
        assert self.input_column, "Input column is not set"
        assert self.output_column, "Output column is not set"

        # BGE Embedding tokenizer will automatically add bos and eos
        # # Add bos and eos token
        # for idx, sample in enumerate(examples[self.input_column]):
        #     examples[self.input_column][idx] = self.tokenizer.special_tokens_map['bos_token'] + sample + self.tokenizer.special_tokens_map['eos_token']
        #     # examples[self.output_column][idx] = '<|im_start|>' + sample + '<|im_end|>'

        tokenized_input = self.tokenizer(examples[self.input_column], truncation=True, padding=True, return_tensors='pt')
        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input["attention_mask"]
       
        return {
            "text": examples["text"],
            "input_ids": input_ids,
            "labels": examples[self.output_column],
            "attention_mask": attention_mask
        }

    # def tokenize_function(self, examples):
    #     assert self.input_column, "Input column is not set"
    #     assert self.output_column, "Output column is not set"

    #     # Add bos and eos token
    #     for idx, sample in enumerate(examples[self.output_column]):
    #         examples[self.output_column][idx] = self.tokenizer.special_tokens_map['cls_token'] + sample + self.tokenizer.special_tokens_map['sep_token']

    #     input = self.tokenizer(examples[self.input_column], truncation=True, padding=True, return_tensors='pt')['input_ids']
    #     label = self.tokenizer(examples[self.output_column], truncation=True, padding=True, return_tensors='pt')['input_ids']
    #     input_sequence = torch.concat([input, label], dim=1)
    #     output_sequence = torch.concat([torch.full_like(input, -100), label], dim=1)


    #     return {
    #         "input_ids": input_sequence,
    #         "labels": output_sequence
    #     }

