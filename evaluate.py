import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import CharacterTokenizer, TransliterationDataset, collate_fn
from utils.metrics import calculate_metrics

def run_evaluation_pipeline(model_type, checkpoint_path, test_data_path, device):
    if not os.path.exists(checkpoint_path):
        print(f"[-] Checkpoint missing for {model_type} at: {checkpoint_path}")
        return

    # 1. Load the checkpoint states
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    # 2. Rehydrate Tokenizers directly using your dataset.py class definitions
    src_tokenizer = CharacterTokenizer(is_target=False)
    tgt_tokenizer = CharacterTokenizer(is_target=True)
    
    # Inject saved vocab maps back into your tokenizer objects
    src_tokenizer.char2idx = checkpoint["src_vocab"]
    tgt_tokenizer.char2idx = checkpoint["tgt_vocab"]
    src_tokenizer.idx2char = {v: k for k, v in src_tokenizer.char2idx.items()}
    tgt_tokenizer.idx2char = {v: k for k, v in tgt_tokenizer.char2idx.items()}

    # 3. Instantiate Data Split Loaders
    test_dataset = TransliterationDataset(test_data_path, src_tokenizer, tgt_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 4. Reconstruct structural network blueprints
    if model_type == "vanilla":
        from models.vanilla_seq2seq import Encoder, Decoder, VanillaSeq2Seq
        encoder = Encoder(len(src_tokenizer.char2idx), config["embedding_dim"], config["hidden_dim"], config["cell_type"], config["num_layers"], dropout=0.0)
        decoder = Decoder(len(tgt_tokenizer.char2idx), config["embedding_dim"], config["hidden_dim"], config["cell_type"], config["num_layers"], dropout=0.0)
        model = VanillaSeq2Seq(encoder, decoder).to(device)
    elif model_type == "attention":
        from models.attention_seq2seq import Encoder, Decoder, AttentionSeq2Seq
        encoder = Encoder(len(src_tokenizer.char2idx), config["embedding_dim"], config["hidden_dim"], config["cell_type"], config["num_layers"], dropout=0.0)
        decoder = Decoder(len(tgt_tokenizer.char2idx), config["embedding_dim"], config["hidden_dim"], config["cell_type"], config["attention_type"], config["num_layers"], dropout=0.0)
        model = AttentionSeq2Seq(encoder, decoder).to(device)

    # Apply trained weights
    model.encoder.load_state_dict(checkpoint["encoder_state"])
    model.decoder.load_state_dict(checkpoint["decoder_state"])
    model.eval()

    # 5. Iterative Greedy Inference Execution
    all_predicted_strings = []
    all_true_strings = []
    
    with torch.no_grad():
        for src, tgt in tqdm(test_loader, desc=f"Running Test Inference [{model_type.upper()}]"):
            src, tgt = src.to(device), tgt.to(device)
            
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            predictions = output.argmax(dim=-1)
            
            # Use your native tokenizer `.decode` directly!
            for pred_seq, true_seq in zip(predictions, tgt):
                # Clean up sequences starting at position 1 to bypass the <sos> tokens
                pred_str = tgt_tokenizer.decode(pred_seq[1:])
                true_str = tgt_tokenizer.decode(true_seq[1:])
                
                all_predicted_strings.append(pred_str)
                all_true_strings.append(true_str)

    # 6. Metrics Aggregation
    metrics = calculate_metrics(all_predicted_strings, all_true_strings)

    # 7. Print Output Report
    print("\n" + "="*50)
    print(f"TEST RESULTS : {model_type.upper()} SEQ2SEQ")
    print("="*50)
    print(f"Word Accuracy      : {metrics['word_accuracy']:.2f} %")
    print(f"Character Accuracy : {metrics['char_accuracy']:.2f} %")
    print(f"CER                : {metrics['cer']:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set_path = "data/dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.test.tsv"
    
    # Run evaluations on both architectures using their best stored metrics
    print("[+] Evaluating Vanilla Baseline Checkpoint...")
    run_evaluation_pipeline("vanilla", "predictions_vanilla/best_vanilla_model.pth", test_set_path, device)

    print("[+] Evaluating Attention Baseline Checkpoint...")
    run_evaluation_pipeline("attention", "predictions_attention/best_attention_model.pth", test_set_path, device)