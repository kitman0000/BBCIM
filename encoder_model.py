import torch
from torch import nn

from sentence_transformers import SentenceTransformer

class EmbeddingBasedIntentModel(torch.nn.Module):
    def __init__(self, embedding_model, device) -> None:
        super().__init__()
        self.n_classes = 18
        
        self.embedding = SentenceTransformer(embedding_model, trust_remote_code=True).to(device)
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.n_classes),
        ).to(device)

        
    def forward(self, input_ids, attention_mask):
        x = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        x = self.embedding(x)["sentence_embedding"]
        x = self.fc(x)

        return x
    

if __name__ == "__main__":
    model_path = "/data/models/BAAI/bge-m3"
    model = EmbeddingBasedIntentModel(model_path, "cuda:0")

    output = model("你好")
    print(output)

