# export PYTHONPATH=./
# accelerate launch ./src/os_intent/train.py

import os
import json

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType
from datasets import load_dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer_utils import set_seed
from torch.optim.adamw import AdamW
from tqdm.auto import tqdm
import datasets
import transformers
from peft import LoraConfig, get_peft_model, TaskType
from peft.optimizers import create_loraplus_optimizer
import swanlab

from accelerate import DistributedDataParallelKwargs

from src.os_intent.encoder_model import EmbeddingBasedIntentModel
from src.os_intent.v1.tokenizer import Tokenizer
from src.os_intent.v1.intent_dataset import get_dataset_loader
from src.os_intent.v1.data_loader import label_dict_id_to_label
from datetime import datetime

use_lora = False

def save_model(model, model_save_path, accelerator, loss=None):
    model_to_save = accelerator.unwrap_model(model) 
    if use_lora:
        model_to_save.save_pretrained(model_save_path + "_best")
    else:
        os.makedirs(model_save_path, exist_ok=True)

        if hasattr(model, "module"):
            _model = model.module
        else:
            _model = model

        torch.save(_model.state_dict(), os.path.join(model_save_path, "model.pth"))

    if loss != None:
        with open(os.path.join(model_save_path, "loss.txt"), "w") as f:
            f.write(str(loss))


def train(model, tokenizer, train_dataloader, eval_dataloader, model_save_path, num_epochs=50, lr=7e-5, seed=1234):
    assert not os.path.exists(model_save_path), f"Model ckpt already exist: {model_save_path}"

    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    if accelerator.is_main_process:
        swanlab.init(
            # 设置项目名
            project="os_intent",
            # 设置超参数
            config={
                "model_save_path": model_save_path,
                "architecture": "bge embedding",
                "dataset": "CIFAR-100",
                "epochs": num_epochs,
                "lr": lr
            }
        )



    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # The seed need to be set before we instantiate the model, as it will determine the random head.
    set_seed(seed)

    # Instantiate optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose="True"
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Instantiate a progress bar to keep track of training. Note that we only enable it on the main
    # process to avoid having 8 progress bars.
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)), disable=not accelerator.is_main_process)
    # Now we train the model

    best_loss = 1
    top_acc = 0
    for epoch in range(num_epochs):
        model.train()
        loss_list = []
        for step, batch in enumerate(train_dataloader):
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(outputs, batch["labels"])
            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            loss_list.append(loss)

        model.eval()
        avg_loss = sum(loss_list) / len(loss_list)
        if accelerator.is_main_process:
            print(f"Epoch: {epoch} Loss: {avg_loss}")
        scheduler.step(avg_loss)

        model.eval()

        total_correct = 0
        total = 0

        class_correct_dict = {}
        class_total_dict = {}

        if epoch % 10 == 0:
            print("Evaluating")
            for step, batch in tqdm(enumerate(eval_dataloader)):
                outputs = model(batch['input_ids'], batch['attention_mask'])
                batch_predicted = outputs.argmax(1)
                correct = (batch['labels'] == batch_predicted).sum().detach().item()
                total_correct += correct
                total += outputs.shape[0]

                for label, predicted in zip(batch["labels"].reshape(-1), batch_predicted):
                    label = label.detach().item()
                    label_name = label_dict_id_to_label[str(label)]
                    if label_name not in class_correct_dict:
                        class_correct_dict[label_name] = 0
                        class_total_dict[label_name] = 0

                    class_total_dict[label_name] += 1
                    if label == predicted:
                        class_correct_dict[label_name] += 1

            acc = total_correct / total
            if accelerator.is_main_process:
                print(f"Acc: {acc}")

                for label in class_correct_dict.keys():
                    print(f"{label}: {class_correct_dict[label] / class_total_dict[label]}")
                swanlab.log({"acc": acc, "loss": avg_loss, "lr": optimizer.param_groups[0]['lr']})

            # Save best checkpoint
            if accelerator.is_main_process:
                if epoch == 0 or avg_loss < best_loss:
                    best_loss = avg_loss
                    print("Saving best model")
                    save_model(model, model_save_path + "_best", accelerator, best_loss)
                else:
                    print(f"best loss: {best_loss}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_model(model, model_save_path + "_final", accelerator)


    # print("Testing:")
    # for batch in eval_dataloader:
    #     test_input = tokenizer.tokenizer(batch['input'][0],return_tensors='pt')['input_ids']
    #     output = tokenizer.tokenizer.batch_decode(model.generate(test_input, max_new_tokens=128), skip_special_tokens=True)

    #     output = output[0][len(batch['input'][0]):]
    #     print(batch['input'])
    #     print(output)
    #     print("")

if __name__ == '__main__':
    device = "cuda"
    model_checkpoint = "/data/models/BAAI/bge-m3"
    model = EmbeddingBasedIntentModel(model_checkpoint, device)

    tokenizer = Tokenizer(AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True))
    multi_gpus = True

    if multi_gpus:
        torch.distributed.init_process_group(
            "nccl",
            init_method=None,
            world_size=-1,
            rank=-1,
            store=None,
            group_name='default',
        )
    batch_size = 32
    num_epochs = 500
    use_dataset_cache = True
    ckpt_path = f"os_intent_ckpts/intent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    test_dataset_path = 'data/amazon_massive_scenario_zh-CN/cleaned_test.jsonl'
    train_dataset_path_list = ['data/amazon_massive_scenario_zh-CN/cleaned_train.jsonl']

    # Load dataset
    train_dataloader, eval_dataloader = get_dataset_loader(tokenizer, 
                                                           batch_size=batch_size, 
                                                           train_dataset_path_list=train_dataset_path_list, 
                                                           test_dataset_path=test_dataset_path, 
                                                           use_cache=use_dataset_cache,
                                                           multi_gpus=multi_gpus)

    train(model, tokenizer, train_dataloader, eval_dataloader, num_epochs=num_epochs, model_save_path=ckpt_path, lr=1e-4)

