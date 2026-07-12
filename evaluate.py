import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import (
    CharacterTokenizer,
    TransliterationDataset,
    collate_fn,
)
from utils.metrics import calculate_metrics
from utils.inference import (
    greedy_decode_attention,
    greedy_decode_vanilla,
    decode_until_eos,
)

MAX_DECODE_LEN = 50

def run_evaluation_pipeline(
    model_type,
    checkpoint_path,
    test_data_path,
    device,
):
    if not os.path.exists(checkpoint_path):
        print(
            f"[-] Checkpoint missing for {model_type}: "
            f"{checkpoint_path}"
        )
        return

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    config = checkpoint["config"]

    # --------------------------------------------------
    # Restore tokenizers
    # --------------------------------------------------

    src_tokenizer = CharacterTokenizer(is_target=False)
    tgt_tokenizer = CharacterTokenizer(is_target=True)

    src_tokenizer.char2idx = checkpoint["src_vocab"]
    tgt_tokenizer.char2idx = checkpoint["tgt_vocab"]

    src_tokenizer.idx2char = {
        idx: char
        for char, idx in src_tokenizer.char2idx.items()
    }

    tgt_tokenizer.idx2char = {
        idx: char
        for char, idx in tgt_tokenizer.char2idx.items()
    }

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    test_dataset = TransliterationDataset(
        test_data_path,
        src_tokenizer,
        tgt_tokenizer,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------
    # Reconstruct model
    # --------------------------------------------------

    if model_type == "vanilla":

        from models.vanilla_seq2seq import (
            Encoder,
            Decoder,
            VanillaSeq2Seq,
        )

        encoder = Encoder(
            len(src_tokenizer.char2idx),
            config["embedding_dim"],
            config["hidden_dim"],
            config["cell_type"],
            config["num_layers"],
            dropout=0.0,
        )

        decoder = Decoder(
            len(tgt_tokenizer.char2idx),
            config["embedding_dim"],
            config["hidden_dim"],
            config["cell_type"],
            config["num_layers"],
            dropout=0.0,
        )

        model = VanillaSeq2Seq(
            encoder,
            decoder,
        ).to(device)

    elif model_type == "attention":

        from models.attention_seq2seq import (
            Encoder,
            Decoder,
            AttentionSeq2Seq,
        )

        encoder = Encoder(
            len(src_tokenizer.char2idx),
            config["embedding_dim"],
            config["hidden_dim"],
            config["cell_type"],
            config["num_layers"],
            dropout=0.0,
        )

        decoder = Decoder(
            len(tgt_tokenizer.char2idx),
            config["embedding_dim"],
            config["hidden_dim"],
            config["cell_type"],
            config["attention_type"],
            config["num_layers"],
            dropout=0.0,
        )

        model = AttentionSeq2Seq(
            encoder,
            decoder,
        ).to(device)

    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    model.encoder.load_state_dict(
        checkpoint["encoder_state"]
    )

    model.decoder.load_state_dict(
        checkpoint["decoder_state"]
    )

    model.eval()

    # --------------------------------------------------
    # Independent greedy inference
    # --------------------------------------------------

    sos_idx = tgt_tokenizer.char2idx[
        tgt_tokenizer.sos_token
    ]

    eos_idx = tgt_tokenizer.char2idx[
        tgt_tokenizer.eos_token
    ]

    all_predicted_strings = []
    all_true_strings = []

    with torch.no_grad():

        for src, tgt in tqdm(
            test_loader,
            desc=f"Test Inference [{model_type.upper()}]",
        ):
            src = src.to(device)
            tgt = tgt.to(device)

            if model_type == "vanilla":

                predictions = greedy_decode_vanilla(
                    model,
                    src,
                    sos_idx,
                    eos_idx,
                    MAX_DECODE_LEN,
                )

            else:

                predictions = greedy_decode_attention(
                    model=model,
                    src=src,
                    sos_idx=sos_idx,
                    eos_idx=eos_idx,
                    max_len=MAX_DECODE_LEN,
                )

            for pred_seq, true_seq in zip(
                predictions,
                tgt,
            ):
                pred_str = tgt_tokenizer.decode(pred_seq)
                true_str = tgt_tokenizer.decode(true_seq)
                # pred_str = decode_until_eos(
                #     pred_seq,
                #     tgt_tokenizer,
                # )

                # # true_seq contains <sos> at index 0.
                # true_str = decode_until_eos(
                #     true_seq[1:],
                #     tgt_tokenizer,
                # )

                all_predicted_strings.append(
                    pred_str
                )

                all_true_strings.append(
                    true_str
                )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    metrics = calculate_metrics(
        all_predicted_strings,
        all_true_strings,
    )

    print("\n" + "=" * 60)

    print(
        f"TEST RESULTS: {model_type.upper()} SEQ2SEQ"
    )

    print("=" * 60)

    print(
        f"Exact Word Accuracy          : "
        f"{metrics['word_accuracy']:.2f} %"
    )

    print(
        f"Positional Character Accuracy: "
        f"{metrics['char_accuracy']:.2f} %"
    )

    print(
        f"Character Error Rate (CER)   : "
        f"{metrics['cer']:.4f}"
    )

    print("=" * 60 + "\n")


if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    test_set_path = (
        "data/dakshina_dataset_v1.0/te/"
        "lexicons/te.translit.sampled.test.tsv"
    )

    print(
        "[+] Evaluating Vanilla Baseline Checkpoint..."
    )

    run_evaluation_pipeline(
        "vanilla",
        "predictions_vanilla/best_vanilla_model_2.pth",
        test_set_path,
        device,
    )

    print(
        "[+] Evaluating Attention Checkpoint..."
    )

    run_evaluation_pipeline(
        "attention",
        "predictions_attention/best_attention_model.pth",
        test_set_path,
        device,
    )