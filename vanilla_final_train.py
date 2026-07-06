import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import time
from utils.dataset import CharacterTokenizer, TransliterationDataset, collate_fn
from models.vanilla_seq2seq import Encoder, Decoder, VanillaSeq2Seq

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
            if char_t == 0: # pad token , break
                break
            if char_p != char_t: # predicted wrong char
                match = False
                break
        if match:
            correct_words += 1
            
    return correct_words / total_words

# character level
def calculate_char_accuracy(model_outputs, target_tensors):
    predictions = model_outputs.argmax(dim = -1)

    pred_words = predictions[:, 1:]
    true_words = target_tensors[:,1:]

    correct = 0
    total = 0

    for p, t in zip(pred_words, true_words):
        for char_p, char_t in zip(p, t):
            if char_t == 0: 
                break

            total += 1
            if char_p == char_t: 
                correct += 1

    return correct / total

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_word_acc = 0
    epoch_char_acc = 0
    
    for src, tgt in loader:
        src, tgt = src.to(device, non_blocking = True), tgt.to(device, non_blocking = True)
        
        optimizer.zero_grad()
        output = model(src, tgt, teacher_forcing_ratio=0.5)
        
        # currently o.p - (bs, maxlen, vocab dim) we need to convert to (bs*maxlen, vocab dim) and tgt also to - (bs*maxlen)
        # tgt only has index at each time step and and op has logits of all vocab at every time step
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
            src, tgt = src.to(device, non_blocking = True), tgt.to(device, non_blocking = True)
            output = model(src, tgt, teacher_forcing_ratio=0.0) # no teacher forcing in validation
            
            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, output_dim), tgt[:, 1:].reshape(-1))
            
            epoch_loss += loss.item()
            epoch_word_acc += calculate_accuracy(output, tgt)
            epoch_char_acc += calculate_char_accuracy(output, tgt)
            
    return epoch_loss / len(loader), epoch_word_acc / len(loader), epoch_char_acc / len(loader) 

def run_final():
    # config
    best_config = {
        "embedding_dim": 64,
        "hidden_dim": 512,
        "cell_type": "LSTM",
        "num_layers": 2,
        "dropout": 0.4, # 0.2, 0.1 hadhnt worked well 
        "learning_rate": 1e-3,
        "weight_decay": 1e-5, # was overfitting , so added it
        "epochs": 20
    }

    wandb.init(
        project="telugu-vanilla-seq2seq", 
        name="final-optimized-vanilla",
        config=best_config
    )

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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True)
    
    # model intialisation
    encoder = Encoder(len(src_tokenizer.char2idx), config.embedding_dim, config.hidden_dim, config.cell_type, config.num_layers, config.dropout)
    decoder = Decoder(len(tgt_tokenizer.char2idx), config.embedding_dim, config.hidden_dim, config.cell_type, config.num_layers, config.dropout)
    model = VanillaSeq2Seq(encoder, decoder).to(device)
    
    # loss, algo
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",factor=0.5, patience=2)

    # tracking
    best_val_acc = -1
    patience = 5
    patience_counter = 0
    best_epoch = -1

    start_time = time.time()
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
            "val_char_accuracy": val_char_acc,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Word Acc: {val_word_acc:.4f}")
        scheduler.step(val_word_acc)

        # early stopping
        if val_word_acc > best_val_acc:
            best_val_acc = val_word_acc
            patience_counter = 0
            best_epoch = epoch+1
            
            if os.path.exists("/content/drive"):
                drive_save_dir = "/content/drive/MyDrive/Seq2Seq_Project/predictions_vanilla"
            else:
                drive_save_dir = "predictions_vanilla"
                
            os.makedirs(drive_save_dir, exist_ok=True)
            checkpoint_path = os.path.join(drive_save_dir, "best_vanilla_model.pth")

            checkpoint = {
                "epoch": epoch,

                "best_val_accuracy": best_val_acc,

                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),

                "optimizer_state": optimizer.state_dict(),

                "src_vocab": src_tokenizer.char2idx,
                "tgt_vocab": tgt_tokenizer.char2idx,

                "config": best_config,
                "learning_rate": optimizer.param_groups[0]["lr"]
            }
            torch.save(checkpoint, checkpoint_path)
            print("Val_loss imporved , checkpoint saved.")
        else:
            patience_counter += 1
            print(f"Val acc did not improve. Early stopping : {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print("Early stopping. Terminating run.")
            break
        
    training_time = time.time() - start_time
    wandb.run.summary["training_time_minutes"] = training_time / 60
    wandb.run.summary["best_val_word_accuracy"] = best_val_acc
    wandb.run.summary["best_epoch"] = best_epoch
    wandb.finish()


if __name__ == "__main__":
    import pandas as pd
    run_final()