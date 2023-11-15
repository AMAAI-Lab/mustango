import json
import time
import math
import random
import pickle
import gc, os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoTokenizer
from modelling_deberta_v2 import DebertaV2ForTokenClassificationRegression

from transformers import AutoConfig, get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import Adafactor, get_scheduler

from sklearn.metrics import accuracy_score, classification_report, f1_score


class dotdict(dict):
    """Dot notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
class BeatDataset(Dataset):
    def __init__(self, filename, samples=-1):
        
        data = [json.loads(line) for line in open(filename).readlines()]
        if samples != -1:
            data = data[:samples]
        inputs, beat_time, beat_class = [], [], []
        
        for instance in tqdm(data):        
            inputs.append(instance["captions"])
            beat_time.append([round(item, 4) for item in instance["beats"][0]])
            beat_class.append([int(item)-1 for item in instance["beats"][1]])
                
        self.inputs, self.beat_time, self.beat_class = inputs, beat_time, beat_class
        print ("Instances in {}: {}".format(filename, len(self.inputs)))
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.beat_time[index], self.beat_class[index]
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
def configure_dataloaders(train_batch_size=16, eval_batch_size=16, samples=-1):
    """Prepare dataloaders"""
    train_dataset = BeatDataset("data/100train_combined_chatgpt_aug4.json", samples)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size, collate_fn=train_dataset.collate_fn)

    val_dataset = BeatDataset("data/100eval_combined_chatgpt_aug4.json", samples)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=eval_batch_size, collate_fn=val_dataset.collate_fn)

    test_dataset = BeatDataset("data/100test_musiccaps_ep4.json", samples)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=eval_batch_size, collate_fn=val_dataset.collate_fn)
    
    return train_loader, val_loader, test_loader


def configure_optimizer(model, args):
    """Prepare optimizer and schedule (linear warmup and decay)"""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    return optimizer


def configure_scheduler(optimizer, num_training_steps, args):
    """Prepare scheduler"""
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else math.ceil(num_training_steps * args.warmup_ratio)
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )    
    return lr_scheduler

        
def train_or_eval_model(model, tokenizer, dataloader, optimizer=None, split="Train"):
    """Training and evaluation loop"""
    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    cls_losses, reg_losses, all_preds, all_labels = [], [], [], []
    instance_beats, instance_timestamps = [], []
    if split == "Train":
        model.train()
    else:
        model.eval()
    
    for batch in tqdm(dataloader, leave=False):
        
        texts, timestamps, labels =  batch
        tokenized = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
        token_lengths = [tokenized["input_ids"].shape[1] for _ in range(len(texts))]
        for k, item in enumerate(tokenized["input_ids"]):
            index = torch.where(item == 0)[0]
            if len(index) != 0:
                token_lengths[k] = int(index[0].item())

        new_labels = [labels[k][:token_lengths[k]] for k in range(len(texts))]
        new_timestamps = [timestamps[k][:token_lengths[k]] for k in range(len(texts))]
        beat_lengths = [len(item) for item in new_labels]

        flat_labels = torch.tensor([item for sublist in new_labels for item in sublist], dtype=torch.int64, device=model.device)
        flat_timestamps = torch.tensor([item for sublist in new_timestamps for item in sublist], device=model.device)
        
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
        
        if split == "Train":
            out = model(**tokenized)
        else:
            with torch.no_grad():
                out = model(**tokenized)
                
        beat_logits = torch.cat([out["logits"][k, :beat_lengths[k]] for k in range(len(texts))])        
        beat_times = torch.cat([out["values"][k, :beat_lengths[k]] for k in range(len(texts))]).flatten()

        loss1 = cross_entropy_loss_fn(beat_logits, flat_labels)
        loss2 = mse_loss_fn(beat_times, flat_timestamps)
        loss = loss1 + loss2
        
        if split == "Train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        cls_losses.append(loss1.item())
        reg_losses.append(loss2.item())        
        all_preds += torch.argmax(beat_logits, 1).cpu().numpy().tolist()
        all_labels += flat_labels.cpu().numpy().tolist()
             
        predicted_class = torch.argmax(out["logits"], -1).detach().cpu().numpy().astype("int32").tolist()
        predicted_times = out["values"][:, :, 0].detach().cpu().numpy().astype("float32").round(4).tolist()

        instance_beats += [list(item[:token_lengths[k]]) for k, item in enumerate(predicted_class)]
        instance_timestamps += [list(item[:token_lengths[k]]) for k, item in enumerate(predicted_times)]

    avg_cls_loss = round(np.mean(cls_losses), 4)
    avg_reg_loss = round(np.mean(reg_losses), 4)
    
    all_preds = np.array(all_preds) + 1
    all_labels = np.array(all_labels) + 1
    acc = round(100 * accuracy_score(all_labels, all_preds), 2)
    f1 = round(100 * f1_score(all_labels, all_preds, average="weighted"), 2)
    cls_report = classification_report(all_labels, all_preds)
    
    return avg_cls_loss, avg_reg_loss, acc, f1, cls_report, instance_beats, instance_timestamps
    
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate for transformers.")
    parser.add_argument("--wd", default=0.0, type=float, help="Weight decay for transformers.")
    parser.add_argument("--warm-up-steps", type=int, default=0, help="Warm up steps.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
    parser.add_argument("--bs", type=int, default=8, help="Batch size.")
    parser.add_argument("--eval-bs", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs.")
    parser.add_argument("--name", default="microsoft/deberta-v3-large", help="Which model.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument('--save', action='store_true', default=False, help="Save best model weights.")
    parser.add_argument("--samples", type=int, default=-1, help="Samples to use for training and evaluation.")
    
    global args
    global evaluator
    args = parser.parse_args()
    print(args)
    
    train_batch_size = args.bs
    eval_batch_size = args.eval_bs
    epochs = args.epochs
    name = args.name
    save = args.save
    samples = args.samples
    
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = DebertaV2ForTokenClassificationRegression.from_pretrained(name).cuda()
    print ("Num trainable parameters in model: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    train_loader, val_loader, test_loader = configure_dataloaders(train_batch_size, eval_batch_size, samples)
    optimizer = configure_optimizer(model, args)
    
    exp_id = str(int(time.time()))
    vars(args)["exp_id"] = exp_id
    
    path = "saved/beats-model/" + exp_id + "/" + name.replace("/", "-")
    Path("saved/beats-model/" + exp_id + "/").mkdir(parents=True, exist_ok=True)
    
    fname = "saved/beats-model/" + exp_id + "/" + "args.txt"
    
    f = open(fname, "a")
    f.write(str(args) + "\n\n")
    f.close()
        
    lf_name = "results/beats-model/" + name.replace("/", "-") + ".txt"
    lf_buffer = str(args) + "\n\n"
    
    best_val_score = 0
    
    for e in range(epochs):
        train_cls_loss, train_reg_loss, train_acc, train_f1, train_cls_report, train_beats, train_timestamps = train_or_eval_model(model, tokenizer, train_loader, optimizer, "Train")
        val_cls_loss, val_reg_loss, val_acc, val_f1, val_cls_report, val_beats, val_timestamps = train_or_eval_model(model, tokenizer, val_loader, None, "Dev")
        test_cls_loss, test_reg_loss, test_acc, test_f1, test_cls_report, test_beats, test_timestamps = train_or_eval_model(model, tokenizer, test_loader, None, "Test")
        
        if save:
            if val_iou > best_val_score:
                torch.save(model.state_dict(), path + "-best-val-iou.pt")
                best_val_score = val_f1
        
        x1 = "Epoch {}: Classification Loss: Train {}, Val {}, Test {}; Regression Loss: Train {}, Val {}, Test {}".format(e+1, train_cls_loss, val_cls_loss, test_cls_loss, train_reg_loss, val_reg_loss, test_reg_loss)
        x2 = "Accuracy: Train {}, Val {}, Test {}; F1: Train {}, Val {}, Test {}".format(train_acc, val_acc, test_acc, train_f1, val_f1, test_f1)
        x = "\n".join([x1, x2]) + "\n\nValid Classification Report:\n" + val_cls_report + "\n\nTest Classification Report:\n" + test_cls_report
        print (x)
        
        with open("saved/beats-model/" + exp_id + "/val_preds_epoch_{}.json".format(e+1), "w") as f:
            for beat_class, beat_time in zip(val_beats, val_timestamps):
                rounded_beat_time = [round(item, 4) for item in beat_time]
                f.write(json.dumps({"beats": [beat_class, rounded_beat_time]}) + "\n")
                
        with open("saved/beats-model/" + exp_id + "/test_preds_epoch_{}.json".format(e+1), "w") as f:
            for beat_class, beat_time in zip(test_beats, test_timestamps):
                rounded_beat_time = [round(item, 4) for item in beat_time]
                f.write(json.dumps({"beats": [beat_class, rounded_beat_time]}) + "\n")
        
        lf_buffer += x + "\n\n"

        f = open(fname, "a")
        f.write(x + "\n\n")
        f.close()
        
    lf = open(lf_name, "a")
    lf.write(lf_buffer + "-"*100 + "\n")
    lf.close()