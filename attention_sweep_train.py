import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import pandas as pd

from utils.dataset import CharacterTokenizer, TransliterationDataset, collate_fn
from models.attention_seq2seq import Encoder, Decoder, AttentionSeq2Seq

# word level (primary metric)
def calculate_accuracy(model_outputs, target_tensors):
    predictions = model_outputs.argmax(dim=-1)
    pred_words = predictions[:, 1:]
    true_words = target_tensors[:, 1:]
    
    correct_words = 0
    total_words = target_tensors.shape[0]
    
    for p, t in zip(pred_words, true_words):
        match = True
        for char_p, char_t in zip(p, t):
            if char_t == 0:  
                break
            if char_p != char_t:  
                match = False
                break
        if match:
            correct_words += 1
            
    return correct_words / total_words

# Character level 
def calculate_char_accuracy(model_outputs, target_tensors):
    predictions = model_outputs.argmax(dim=-1)

    pred_words = predictions[:, 1:]
    true_words = target_tensors[:, 1:]

    correct = 0
    total = 0

    for p, t in zip(pred_words, true_words):
        for char_p, char_t in zip(p, t):
            if char_t == 0: 
                break

            total += 1
            if char_p == char_t: 
                correct += 1

    return correct / total if total > 0 else 0.0

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_word_acc = 0
    epoch_char_acc = 0
    
    for src, tgt in loader:
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(src, tgt, teacher_forcing_ratio=0.5)
        
        output_dim = output.shape[-1]
        loss = criterion(output[:, 1:].reshape(-1, output_dim), tgt[:, 1:].reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_word_acc += calculate_accuracy(output, tgt)
        epoch_char_acc += calculate_char_accuracy(output, tgt)
        
    return epoch_loss / len(loader), epoch_word_acc / len(loader), epoch_char_acc / len(loader) 

def validate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_word_acc = 0
    epoch_char_acc = 0
    
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            output = model(src, tgt, teacher_forcing_ratio=0.0) 
            
            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, output_dim), tgt[:, 1:].reshape(-1))
            
            epoch_loss += loss.item()
            epoch_word_acc += calculate_accuracy(output, tgt)
            epoch_char_acc += calculate_char_accuracy(output, tgt)
            
    return epoch_loss / len(loader), epoch_word_acc / len(loader), epoch_char_acc / len(loader) 

def run_sweep():
    wandb.init()
    config = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # paths
    train_path = "data/dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.train.tsv"
    dev_path = "data/dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.dev.tsv"
    
    # vocab build
    df_train = pd.read_csv(train_path, sep="\t", names=["native", "romanized", "count"], header=None).dropna()
    src_tokenizer = CharacterTokenizer(is_target=False)
    tgt_tokenizer = CharacterTokenizer(is_target=True)
    src_tokenizer.build_vocab(df_train["romanized"].astype(str).tolist())
    tgt_tokenizer.build_vocab(df_train["native"].astype(str).tolist())
    
    # dataset class
    train_dataset = TransliterationDataset(train_path, src_tokenizer, tgt_tokenizer)
    dev_dataset = TransliterationDataset(dev_path, src_tokenizer, tgt_tokenizer)
    
    # loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)
    
    # model intialisation
    encoder = Encoder(
        vocab_size=len(src_tokenizer.char2idx), 
        embedding_dim=config.embedding_dim, 
        hidden_dim=config.hidden_dim, 
        cell_type=config.cell_type, 
        num_layers=config.num_layers, 
        dropout=config.dropout
    )
    
    decoder = Decoder(
        vocab_size=len(tgt_tokenizer.char2idx), 
        embedding_dim=config.embedding_dim, 
        hidden_dim=config.hidden_dim, 
        cell_type=config.cell_type, 
        attention_type=config.attention_type,  # Sweeps will explore 'bahdanau' or 'luong'
        num_layers=config.num_layers, 
        dropout=config.dropout
    )
    
    model = AttentionSeq2Seq(encoder, decoder).to(device)
    
    #loss,algo
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignoring <pad> elements (index 0)
    
    # track
    best_val_acc = -1.0

    for epoch in range(config.epochs):
        train_loss, train_word_acc, train_char_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_word_acc, val_char_acc = validate(model, dev_loader, criterion, device)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_word_accuracy": train_word_acc,
            "train_char_accuracy": train_char_acc,
            "val_loss": val_loss,
            "val_word_accuracy": val_word_acc,
            "val_char_accuracy": val_char_acc
        })

        if val_word_acc > best_val_acc:
            best_val_acc = val_word_acc

            if os.path.exists("/content/drive"):
                drive_save_dir = "/content/drive/MyDrive/Seq2Seq_Project/predictions_attention"
            else:
                drive_save_dir = "predictions_attention"
                
            os.makedirs(drive_save_dir, exist_ok=True)
            checkpoint_path = os.path.join(drive_save_dir, "best_attention_model.pth")

            checkpoint = {
                "epoch": epoch,
                "best_val_accuracy": best_val_acc,
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "src_vocab": src_tokenizer.char2idx,
                "tgt_vocab": tgt_tokenizer.char2idx,
                "config": {
                    "embedding_dim": config.embedding_dim,
                    "hidden_dim": config.hidden_dim,
                    "cell_type": config.cell_type,
                    "attention_type": config.attention_type,
                    "num_layers": config.num_layers,
                    "dropout": config.dropout
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Best Attention Model Saved | Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    sweep_config = {
        'method': 'bayes', 
        'metric': {'name': 'val_word_accuracy', 'goal': 'maximize'},
        'parameters': {
            'embedding_dim': {'values': [32, 64, 128]},
            'hidden_dim': {'values': [128, 256, 512]},
            'cell_type': {'values': ['GRU', 'LSTM']},
            'attention_type': {'values': ['bahdanau', 'luong']},
            'num_layers': {'value': 2},
            'dropout': {'values': [0.1, 0.2]},
            'epochs': {'value': 5}
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="telugu-attention-seq2seq")
    wandb.agent(sweep_id, function=run_sweep, count=10)