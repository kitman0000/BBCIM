# BBCIM: BGE-Embedding Based Chinese Intent Model

## Overview
A lightweight intent classification model for chinese. It is designed to be modular, easy to integrate, and optimized for both performance and inference speed.

You can easily influence the model on CPU.

## Key Features
Trained with [SetFit/amazon_massive_scenario_zh-CN](https://huggingface.co/datasets/SetFit/amazon_massive_scenario_zh-CN) dataset 

Uses SentenceTransformer (Sentence-BERT) for high-quality contextual sentence embeddings

Lightweight fully connected head with dropout regularization to prevent overfitting

Optimized for 18 intent classes (easily configurable for other class counts if you want)

Device-agnostic implementation (supports CPU/GPU via PyTorch device configuration)

## Scores:
| Intent         | Accuracy |
|----------------|-------|
| News           | 0.847 |
| Email          | 0.963 |
| IOT            | 0.968 |
| Play           | 0.946 |
| General        | 0.608 |
| Calendar       | 0.925 |
| Weather        | 0.936 |
| QA             | 0.878 |
| Takeway        | 0.895 |
| Lists          | 0.852 |
| Transports     | 0.919 |
| Social         | 0.877 |
| Datetime       | 0.951 |
| Music          | 0.840 |
| Cooking        | 0.847 |
| Alram          | 0.990 |
| Recommendation | 0.830 |
| Audio          | 0.935 |
| Average          | 0.889 |



## Installation
Prerequisites

Python 3.11

PyTorch 2.6.0

SentenceTransformers 5.2.3

accelerate 1.9.0

swanlab

transformers 5.2.0

[BGE M3 Embedding Model](https://huggingface.co/BAAI/bge-m3)

## Usage

Download the checkpoint here [kitman0000/BBCIM](https://huggingface.co/kitman0000/BBCIM)

Use just a few simple lines of codes to inference

```python
from inference import EmbeddingBasedIntentModelWrapper

device = "cpu"
embedding_path = 'YOUR_PATH_TO_BGE_EMBEDDING'
model_checkpoint = "YOUR_PATH_TO_THE_MODEL"

model = EmbeddingBasedIntentModelWrapper(embedding_path, model_checkpoint, device)

while True:
    input_text = input("Enter input: ")
    result = model.classify(input_text)
    print(result)
```

Output:
```
Enter input: 帮我开个灯
iot
Enter input: 青花瓷
play
Enter input: 外面冷不冷
weather
Enter input: 点个汉堡王
takeaway
Enter input: 买张去东京的机票
transport
Enter input: 英国伦敦现在几点
datetime
Enter input: 给谢老板发个邮件
email
Enter input: 提醒我下周六和小王出去玩
calendar
Enter input: 定个明天早上9点的闹钟
alarm
Enter input: 音量调到最小
audio
```
