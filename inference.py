import torch
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
)



from src.os_intent.v1.intent_dataset import get_dataset_loader
from src.os_intent.encoder_model import EmbeddingBasedIntentModel
from src.os_intent.v1.tokenizer import Tokenizer
from src.os_intent.v1.data_loader import label_dict_label_to_id, label_dict_id_to_label

torch.set_printoptions(precision=4,sci_mode=False)

class EmbeddingBasedIntentModelWrapper():
    def __init__(self, embedding_path, model_checkpoint, device):
        self.model = EmbeddingBasedIntentModel(embedding_path, device)
        self.model.load_state_dict(torch.load(model_checkpoint))
        self.model.eval()

        self.tokenizer = Tokenizer(AutoTokenizer.from_pretrained(embedding_path, trust_remote_code=True))

    def classify(self, text):
        test_input = self.tokenizer.tokenizer(text,return_tensors='pt')
        test_input_token = test_input['input_ids'].to(device)

        output = self.model(test_input_token, test_input["attention_mask"])
        output = torch.softmax(output, 1)
        output = label_dict_id_to_label[str(output.argmax(1)[0].detach().item())]

        return output

if __name__ == '__main__':
    eval = False
    device = "cuda:0"
    embedding_path = '/data/models/BAAI/bge-m3'
    model_checkpoint = "os_intent_ckpts/intent_20260320_122206_best/model.pth"

    model = EmbeddingBasedIntentModelWrapper(embedding_path, model_checkpoint, device)

    if eval:

        test_dataset_path = 'data/amazon_massive_scenario_zh-CN/cleaned_test.jsonl'
        train_dataset_path_list = ['data/amazon_massive_scenario_zh-CN/cleaned_train.jsonl']

        # Load dataset
        _, eval_dataloader = get_dataset_loader(model.tokenizer, 
                                                            batch_size=1, 
                                                            train_dataset_path_list=train_dataset_path_list, 
                                                            test_dataset_path=test_dataset_path, 
                                                            use_cache=True,
                                                            multi_gpus=False)
        score_dict = {}
        for data in tqdm(eval_dataloader):
            result = model.classify(data["text"][0])
            gold = data['labels'].detach().item()
            pred = label_dict_label_to_id[result]
            gold_label = label_dict_id_to_label[str(gold)]

            score = score_dict.get(gold_label, {"total": 0, "correct": 0})
            score["total"] += 1
            if gold == pred:
                score["correct"] += 1
            score_dict[gold_label] = score
        
        for item_name, item_score in score_dict.items():
            print(f"{item_name}: {round(item_score['correct']/ item_score['total'], 3)}")
    else:
        while True:
            input_text = input("Enter input: ")
            result = model.classify(input_text)
            print(result)