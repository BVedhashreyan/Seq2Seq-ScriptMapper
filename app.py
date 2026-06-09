import os
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader

from src.dataset import CharacterTokenizer, TransliterationDataset, collate_fn

st.set_page_config(page_title="Telugu Translit Engine", layout="wide")
st.title("🏹 Telugu Script Mapper: Data & Tokenizer Inspector")

# Configure the path to your extracted Telugu files
DATA_DIR = "data/dakshina_dataset_v1.0/te/lexicons/"
TRAIN_FILE = os.path.join(DATA_DIR, "te.translit.sampled.train.tsv")

if not os.path.exists(TRAIN_FILE):
    st.error(f"Dataset not found at `{TRAIN_FILE}`. Please ensure your Telugu files are unpacked correctly.")
else:
    # 1. Preview Raw Data
    raw_df = pd.read_csv(TRAIN_FILE, sep="\t", names=["Telugu Word", "Romanized", "Count"], header=None).dropna()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Raw Data Explorer")
        st.dataframe(raw_df.head(10), use_container_width=True)
        st.metric(label="Total Training Pairs", value=f"{len(raw_df):,}")
        
    # 2. Build and Inspect Tokenizers
    src_tokenizer = CharacterTokenizer(is_target=False)
    tgt_tokenizer = CharacterTokenizer(is_target=True)
    
    src_tokenizer.build_vocab(raw_df["Romanized"].astype(str).tolist())
    tgt_tokenizer.build_vocab(raw_df["Telugu Word"].astype(str).tolist())
    
    with col2:
        st.subheader("🔢 Tokenizer Statistics")
        st.success(f"Source (Latin) Vocab Size: **{len(src_tokenizer.char2idx)}** tokens")
        st.success(f"Target (Telugu) Vocab Size: **{len(tgt_tokenizer.char2idx)}** tokens")
        
        with st.expander("View Telugu Character Map (char2idx)"):
            st.json(tgt_tokenizer.char2idx)

    # 3. Interactive Encoding & Decoding Sandbox
    st.markdown("---")
    st.subheader("🧪 Tokenizer Sandbox")
    st.write("See how a word is processed into a tensor, and how we use `decode` to convert it back.")
    
    user_input = st.text_input("Type a Romanized word to test:", "amma")
    
    if user_input:
        encoded_tensor = src_tokenizer.encode(user_input.strip())
        # Here is where we test your decode function!
        decoded_output = src_tokenizer.decode(encoded_tensor)
        
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Input String:**\n `{user_input}`")
        c2.warning(f"**Generated Tensor:**\n `{encoded_tensor.tolist()}`")
        c3.success(f"**Decoded Word:**\n `{decoded_output}`")

    # 4. Verify Padded Batches
    st.markdown("---")
    st.subheader("📦 Mini-Batch Padding Verification")
    
    dataset = TransliterationDataset(TRAIN_FILE, src_tokenizer, tgt_tokenizer)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    for src_batch, tgt_batch in data_loader:
        st.write(f"Source Batch Shape: `{list(src_batch.shape)}` | Target Batch Shape: `{list(tgt_batch.shape)}`")
        st.write("Visualized Padded Source Matrix (Rows are padded with `0` to match the longest word):")
        st.dataframe(pd.DataFrame(src_batch.numpy()), use_container_width=False)
        break