import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, cell_type="LSTM", num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type.upper()
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if self.cell_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
            
        self.fc_hidden = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim*2, hidden_dim) if cell_type == 'LSTM' else None

    def forward(self, src_batch):
        embedded = self.embedding(src_batch)
        outputs, hidden = self.rnn(embedded)

        if self.cell_type == 'LSTM':
            h, c = hidden

            h = self.fc_hidden(torch.cat((h[0:h.size(0):2], h[1:h.size(0):2]), dim=2))
            c = self.fc_cell(torch.cat((c[0:c.size(0):2], c[1:c.size(0):2]), dim=2))
            hidden = (h, c)
        else:
            hidden = self.fc_hidden(torch.cat((hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]), dim=2))
        
        return outputs, hidden

class Bahdanau(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_a = nn.Linear(hidden_dim*2, hidden_dim, bias= False)
        self.V_a = nn.Linear(hidden_dim, 1, bias = False)

    def forward(self, prev_decoder_hidden, encoder_ouputs, mask = None):
        dec_proj = self.W_a(prev_decoder_hidden).unsqueeze(1)
        enc_proj = self.U_a(encoder_ouputs)

        e = torch.tanh(dec_proj + enc_proj)
        score = self.V_a(e).squeeze(2)

        if mask is not None:
            score = score.masked_fill(~mask, -1e9)
            
        return F.softmax(score, dim=1)

class Luong(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim*2, hidden_dim, bias=False)

    def forward(self, curr_decoder_hidden, encoder_ouputs, mask = None):
        enc_proj = self.W_a(encoder_ouputs)
        q = curr_decoder_hidden.unsqueeze(1)

        score = torch.bmm(q, enc_proj.transpose(1,2)).squeeze(1)

        if mask is not None:
            score = score.masked_fill(~mask, -1e9)

        return F.softmax(score, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, cell_type="LSTM", attention_type="bahdanau", num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type.upper()
        self.num_layers = num_layers
        self.attention_type = attention_type.lower()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if self.attention_type == "bahdanau":
            self.attention = Bahdanau(hidden_dim)
        else:
            self.attention = Luong(hidden_dim)
        
        if self.cell_type == "RNN":
            self.rnn = nn.RNN(embedding_dim + (hidden_dim*2), hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(embedding_dim + (hidden_dim*2), hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim + (hidden_dim*2), hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        
        self.fc_out = nn.Linear(hidden_dim + (hidden_dim * 2) + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, prev_pred, prev_hidden, encoder_outputs, src_mask):
        embedded = self.dropout(self.embedding(prev_pred))

        if self.attention_type == "bahdanau":
            st_1 = prev_hidden[0][-1] if self.cell_type == "LSTM" else prev_hidden[-1]

            attn_weights = self.attention(st_1, encoder_outputs, src_mask)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

            rnn_inp = torch.cat((embedded, context), dim=2)
            ouputs, hidden = self.rnn(rnn_inp, prev_hidden)
        else:
            zero_context = torch.zeros(embedded.shape[0], 1, encoder_outputs.shape[-1]).to(embedded.device)
            rnn_inp = torch.cat((embedded, zero_context), dim=2)
            ouputs, hidden = self.rnn(rnn_inp, prev_hidden)

            st = hidden[0][-1] if self.cell_type == "LSTM" else hidden[-1]

            attn_weights = self.attention(st, encoder_outputs, src_mask)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        output_projection = torch.cat((ouputs, context, embedded), dim=2).squeeze(1)
        predictions = self.fc_out(output_projection)

        return predictions, hidden, attn_weights
            

class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src_batch, tgt_batch, teacher_forcing_ratio=0.5):
        batch_size = src_batch.shape[0]
        max_len = tgt_batch.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, max_len, tgt_vocab_size).to(src_batch.device)
        
        encoder_outputs, encoder_hidden = self.encoder(src_batch)
        
        src_mask = (src_batch != 0)

        decoder_hidden = encoder_hidden
        decoder_input = tgt_batch[:, 0].unsqueeze(1)
        
        for t in range(1, max_len):
            prediction, decoder_hidden, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs,src_mask)
            outputs[:, t, :] = prediction
            
            top_prediction = prediction.argmax(1).unsqueeze(1)
            
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            decoder_input = tgt_batch[:, t].unsqueeze(1) if use_teacher_forcing else top_prediction
            
        return outputs