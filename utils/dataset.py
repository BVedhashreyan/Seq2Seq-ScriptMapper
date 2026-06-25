import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class CharacterTokenizer:
    def __init__(self, is_target=False):
        # is_target = True for telugu and false for english
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"  
        self.eos_token = "<eos>"  
        
        self.special_tokens = [self.pad_token, self.sos_token, self.eos_token]
        self.char2idx = {}
        self.idx2char = {}
        self.is_target = is_target

    def build_vocab(self, words_list):
        # building bidirectional character dictionaries

        # 0-pad, 1-sos, 2-eos
        for idx, token in enumerate(self.special_tokens):
            self.char2idx[token] = idx
            self.idx2char[idx] = token
            
        # extracting unique characters from entire text corpus
        unique_chars = sorted(list(set("".join(words_list))))
        
        next_idx = 3
        for char in unique_chars:
            if char not in self.char2idx:
                self.char2idx[char] = next_idx
                self.idx2char[next_idx] = char
                next_idx += 1
            
        target_script = "Telugu" if self.is_target else "English"
        print(f"Total characters: {len(self.char2idx)} | Target Script: {target_script}")
    
    def encode(self, word):
        # converting word to integer tensor, if telugu then add sos at start
        # Target sequences (Telugu decoders) always start with the <sos> token
        sequence = []
        if self.is_target:
            sequence.append(self.char2idx[self.sos_token])
        
        for char in word:
            # appending character index
            idx = self.char2idx.get(char, self.char2idx[self.pad_token])
            sequence.append(idx)
            
        # appending eos
        sequence.append(self.char2idx[self.eos_token])
        
        return torch.tensor(sequence, dtype=torch.long)

    def decode(self, ids):
        # integers to char
        chars = []
        for token_id in ids:
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()  
                
            char = self.idx2char.get(token_id, " ")
            
            if char in self.special_tokens:
                continue
            chars.append(char)
            
        return "".join(chars)
    
class TransliterationDataset(Dataset):
    def __init__(self, file_path, src_tokenizer, tgt_tokenizer):
        # Read the file. Column 0 is Native (Telugu), Column 1 is Romanized (Latin)
        df = pd.read_csv(file_path, sep="\t", names=["native", "romanized", "count"], header=None)
        df = df.dropna()

        self.src_words = df["romanized"].astype(str).tolist()
        self.tgt_words = df["native"].astype(str).tolist()
        
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
    
    def __len__(self):
        return len(self.src_words)

    def __getitem__(self, idx):
        src_tensor = self.src_tokenizer.encode(self.src_words[idx])
        tgt_tensor = self.tgt_tokenizer.encode(self.tgt_words[idx])
        return src_tensor, tgt_tensor

def collate_fn(batch):
    # padding so that the batch make an even rectangular block
    src_batch, tgt_batch = zip(*batch)
    
    # pad_sequence adds 0 to match the longest item in the batch
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_padded, tgt_padded