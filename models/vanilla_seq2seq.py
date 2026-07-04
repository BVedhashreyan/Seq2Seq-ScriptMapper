import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, cell_type="LSTM", num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()
        
        self.cell_type = cell_type.upper()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if self.cell_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        elif self.cell_type == "GRU":   
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, src_batch):
        embedded = self.embedding(src_batch)

        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, cell_type="LSTM", num_layers=1, dropout=0.0):
        super(Decoder, self).__init__()
        
        self.cell_type = cell_type.upper()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if self.cell_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_char_input, previous_hidden):
        embedded = self.embedding(tgt_char_input)
        outputs, hidden = self.rnn(embedded, previous_hidden)
        
        predictions = self.fc_out(outputs.squeeze(1))
        return predictions, hidden


class VanillaSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_batch, tgt_batch, teacher_forcing_ratio=0.5):
        batch_size = src_batch.shape[0]
        max_len = tgt_batch.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, max_len, tgt_vocab_size).to(src_batch.device)
        
        _, encoder_hidden = self.encoder(src_batch)
        
        decoder_hidden = encoder_hidden
        decoder_input = tgt_batch[:, 0].unsqueeze(1)
        
        for t in range(1, max_len):
            prediction, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t, :] = prediction
            
            top_prediction = prediction.argmax(1).unsqueeze(1)
            
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            decoder_input = tgt_batch[:, t].unsqueeze(1) if use_teacher_forcing else top_prediction
            
        return outputs