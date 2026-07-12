import os
import torch
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nirmala UI", "Gautami", "Arial Unicode MS"]
from utils.dataset import CharacterTokenizer
from utils.inference import (
    greedy_decode_attention,
    decode_until_eos,
)

from models.attention_seq2seq import (
    Encoder,
    Decoder,
    AttentionSeq2Seq,
)

import random


CHECKPOINT_PATH = (
    "predictions_attention/"
    "best_attention_model.pth"
)

SAVE_DIR = "attention_heatmaps"

MAX_DECODE_LEN = 50

def load_test_examples(
    test_path,
    num_examples=10,
    seed=42,
):

    examples = []

    with open(
        test_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            parts = line.strip().split("\t")

            if len(parts) < 2:
                continue

            native_word = parts[0]
            romanized_word = parts[1]

            examples.append(
                (romanized_word, native_word)
            )

    random.seed(seed)

    return random.sample(
        examples,
        min(num_examples, len(examples)),
    )

def load_attention_model(checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    config = checkpoint["config"]

    src_tokenizer = CharacterTokenizer(
        is_target=False
    )

    tgt_tokenizer = CharacterTokenizer(
        is_target=True
    )

    src_tokenizer.char2idx = checkpoint["src_vocab"]

    tgt_tokenizer.char2idx = checkpoint["tgt_vocab"]

    src_tokenizer.idx2char = {
        idx: char
        for char, idx
        in src_tokenizer.char2idx.items()
    }

    tgt_tokenizer.idx2char = {
        idx: char
        for char, idx
        in tgt_tokenizer.char2idx.items()
    }

    encoder = Encoder(
        vocab_size=len(src_tokenizer.char2idx),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        cell_type=config["cell_type"],
        num_layers=config["num_layers"],
        dropout=0.0,
    )

    decoder = Decoder(
        vocab_size=len(tgt_tokenizer.char2idx),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        cell_type=config["cell_type"],
        attention_type=config["attention_type"],
        num_layers=config["num_layers"],
        dropout=0.0,
    )

    model = AttentionSeq2Seq(
        encoder,
        decoder,
    ).to(device)

    model.encoder.load_state_dict(
        checkpoint["encoder_state"]
    )

    model.decoder.load_state_dict(
        checkpoint["decoder_state"]
    )

    model.eval()

    return (
        model,
        src_tokenizer,
        tgt_tokenizer,
    )


def get_source_labels(src_tensor, tokenizer):

    labels = []

    for token_id in src_tensor:

        token_id = token_id.item()

        if token_id == tokenizer.char2idx["<pad>"]:
            continue

        labels.append(
            tokenizer.idx2char[token_id]
        )

    return labels


def get_prediction_labels(
    predicted_tokens,
    tokenizer,
):

    labels = []

    eos_idx = tokenizer.char2idx["<eos>"]

    for token_id in predicted_tokens:

        token_id = token_id.item()

        if token_id == eos_idx:
            labels.append("<eos>")
            break

        if token_id == tokenizer.char2idx["<pad>"]:
            continue

        if token_id == tokenizer.char2idx["<sos>"]:
            continue

        labels.append(
            tokenizer.idx2char[token_id]
        )

    return labels


def generate_attention_heatmap(
    word,
    target_word,
    model,
    src_tokenizer,
    tgt_tokenizer,
    device,
):

    # ----------------------------------------------
    # Encode input
    # ----------------------------------------------

    src = src_tokenizer.encode(
        word.lower()
    ).unsqueeze(0).to(device)

    sos_idx = tgt_tokenizer.char2idx["<sos>"]

    eos_idx = tgt_tokenizer.char2idx["<eos>"]

    # ----------------------------------------------
    # Independent greedy decoding
    # ----------------------------------------------

    with torch.no_grad():

        predictions, attention = (
            greedy_decode_attention(
                model=model,
                src=src,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                max_len=MAX_DECODE_LEN,
                return_attention=True,
            )
        )

    predicted_string = tgt_tokenizer.decode(predictions[0])
    # predicted_string = decode_until_eos(
    #     predictions[0],
    #     tgt_tokenizer,
    # )

    # ----------------------------------------------
    # Build labels
    # ----------------------------------------------

    source_labels = get_source_labels(
        src[0],
        src_tokenizer,
    )

    prediction_labels = get_prediction_labels(
        predictions[0],
        tgt_tokenizer,
    )

    # Number of actual generated decoding steps.
    target_length = len(prediction_labels)

    source_length = len(source_labels)

    attention_matrix = (
        attention[
            0,
            :target_length,
            :source_length,
        ]
        .detach()
        .cpu()
        .numpy()
    )

    # ----------------------------------------------
    # Plot
    # ----------------------------------------------

    fig = plt.figure(figsize=(10, 7))

    ax = fig.add_subplot(111)

    image = ax.imshow(
        attention_matrix,
        aspect="auto",
    )

    ax.set_xticks(
        range(source_length)
    )

    ax.set_xticklabels(
        source_labels,
        rotation=45,
    )

    ax.set_yticks(
        range(target_length)
    )

    ax.set_yticklabels(
        prediction_labels,
    )

    ax.set_xlabel(
        "Romanized Input Characters"
    )

    ax.set_ylabel(
        "Predicted Telugu Characters"
    )

    ax.set_title(
        f"Input: {word}\n"
        f"Target: {target_word} | "
        f"Prediction: {predicted_string}"
    )
    fig.colorbar(
        image,
        ax=ax,
        label="Attention Weight",
    )

    fig.tight_layout()

    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    os.makedirs(
        SAVE_DIR,
        exist_ok=True,
    )

    save_path = os.path.join(
        SAVE_DIR,
        f"{word}_attention.png",
    )

    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"{word:15s} → "
        f"{predicted_string:15s} | "
        f"saved: {save_path}"
    )
    return predicted_string

TEST_PATH = (
    "data/dakshina_dataset_v1.0/"
    "te/lexicons/"
    "te.translit.sampled.test.tsv"
)


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, src_tokenizer, tgt_tokenizer = (
        load_attention_model(
            CHECKPOINT_PATH,
            device,
        )
    )

    examples = load_test_examples(
        TEST_PATH,
        num_examples=10,
        seed=42,
    )

    print("\nGenerating attention heatmaps\n")

    for romanized_word, target_word in examples:

        prediction = generate_attention_heatmap(
            word=romanized_word,
            target_word=target_word,
            model=model,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
        )

        print(
            f"{romanized_word:20s} | "
            f"Target: {target_word:15s} | "
            f"Prediction: {prediction}"
        )


if __name__ == "__main__":
    main()