import torch


def decode_until_eos(token_ids, tokenizer):
    chars = []

    eos_idx = tokenizer.char2idx[tokenizer.eos_token]
    pad_idx = tokenizer.char2idx[tokenizer.pad_token]
    sos_idx = tokenizer.char2idx[tokenizer.sos_token]

    for token_id in token_ids:  
        if isinstance(token_id, torch.Tensor):
            token_id = token_id.item() 

        if token_id == eos_idx:
            break

        if token_id in (pad_idx, sos_idx):
            continue

        chars.append(tokenizer.idx2char.get(token_id, ""))

    return "".join(chars)


def greedy_decode_attention(model, src, sos_idx, eos_idx, max_len=50, return_attention=False):
    encoder_outputs, decoder_hidden = model.encoder(src)

    src_mask = src.ne(0)

    batch_size = src.size(0)

    decoder_input = torch.full(
        (batch_size, 1),
        sos_idx,
        dtype=torch.long,
        device=src.device,
    )

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=src.device,
    )

    generated_tokens = []
    attention_history = []

    for _ in range(max_len):

        prediction, decoder_hidden, attn_weights = model.decoder(
            decoder_input,
            decoder_hidden,
            encoder_outputs,
            src_mask,
        )

        next_token = prediction.argmax(dim=1)

        generated_tokens.append(next_token)

        if return_attention:
            attention_history.append(attn_weights)

        finished |= next_token.eq(eos_idx)

        if finished.all():
            break

        decoder_input = torch.where(
            finished,
            torch.full_like(next_token, eos_idx),
            next_token,
        ).unsqueeze(1)

    generated_tokens = torch.stack(
        generated_tokens,
        dim=1,
    )

    if return_attention:
        attention_matrix = torch.stack(
            attention_history,
            dim=1,
        )

        return generated_tokens, attention_matrix

    return generated_tokens

def greedy_decode_vanilla(model, src, sos_idx, eos_idx, max_len):
    batch_size = src.size(0)

    _, decoder_hidden = model.encoder(src)

    decoder_input = torch.full(
        (batch_size, 1),
        sos_idx,
        dtype=torch.long,
        device=src.device,
    )

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=src.device,
    )

    generated_tokens = []

    for _ in range(max_len):
        prediction, decoder_hidden = model.decoder(
            decoder_input,
            decoder_hidden,
        )

        next_token = prediction.argmax(dim=1)

        generated_tokens.append(next_token)

        finished |= next_token.eq(eos_idx)

        if finished.all():
            break

        # Keep finished sequences at EOS while remaining sequences continue.
        decoder_input = torch.where(
            finished,
            torch.full_like(next_token, eos_idx),
            next_token,
        ).unsqueeze(1)

    return torch.stack(generated_tokens, dim=1)
