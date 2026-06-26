import os
import torch
import pandas as pd
import streamlit as st
from torch.utils.data import DataLoader

from utils.dataset import CharacterTokenizer, TransliterationDataset, collate_fn
from models.vanilla_seq2seq import Encoder, Decoder, VanillaSeq2Seq

# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Telugu ScriptMapper",
    page_icon="🏹",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "data/dakshina_dataset_v1.0/te/lexicons/"
TRAIN_FILE = os.path.join(DATA_DIR, "te.translit.sampled.train.tsv")
WEIGHTS_PATH = "predictions_vanilla/best_vanilla_model.pth"

# A little CSS so Streamlit's defaults feel less like defaults.
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        h1 { letter-spacing: -0.02em; }
        .stMetric { background: rgba(127,127,127,0.06); border-radius: 10px; padding: 0.6rem 0.8rem; }
        .telugu-output {
            font-size: 2rem;
            font-weight: 600;
            padding: 1rem 1.2rem;
            border-radius: 10px;
            background: linear-gradient(135deg, rgba(255,153,51,0.10), rgba(19,136,8,0.10));
            border: 1px solid rgba(127,127,127,0.15);
        }
        .mono-box {
            font-family: "SFMono-Regular", Consolas, monospace;
            background: rgba(127,127,127,0.08);
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", names=["Telugu Word", "Romanized", "Count"], header=None)
    return df.dropna()


@st.cache_resource(show_spinner=False)
def build_tokenizers(latin_words: tuple, telugu_words: tuple):
    src_tok = CharacterTokenizer(is_target=False)
    tgt_tok = CharacterTokenizer(is_target=True)
    src_tok.build_vocab(list(latin_words))
    tgt_tok.build_vocab(list(telugu_words))
    return src_tok, tgt_tok


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Returns (encoder, decoder, src_tokenizer, tgt_tokenizer) or all-None if missing."""
    if not os.path.exists(path):
        return None, None, None, None

    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    cfg = checkpoint["config"]

    src_tok = CharacterTokenizer(is_target=False)
    src_tok.char2idx = checkpoint["src_vocab"]
    src_tok.idx2char = {v: k for k, v in checkpoint["src_vocab"].items()}

    tgt_tok = CharacterTokenizer(is_target=True)
    tgt_tok.char2idx = checkpoint["tgt_vocab"]
    tgt_tok.idx2char = {v: k for k, v in checkpoint["tgt_vocab"].items()}

    encoder = Encoder(len(src_tok.char2idx), cfg["embedding_dim"], cfg["hidden_dim"], cfg["cell_type"], cfg["num_layers"])
    decoder = Decoder(len(tgt_tok.char2idx), cfg["embedding_dim"], cfg["hidden_dim"], cfg["cell_type"], cfg["num_layers"])

    encoder.load_state_dict(checkpoint["encoder_state"])
    decoder.load_state_dict(checkpoint["decoder_state"])
    encoder.eval()
    decoder.eval()

    return encoder, decoder, src_tok, tgt_tok


def transliterate_sentence(sentence: str, encoder, decoder, src_tok, tgt_tok, max_steps: int = 25) -> list[tuple[str, str]]:
    """Returns a list of (input_word, predicted_word) pairs."""
    results = []
    for word in sentence.split():
        clean_word = "".join(c for c in word if c.isalnum()).lower()
        if not clean_word:
            continue

        input_tensor = src_tok.encode(clean_word).unsqueeze(0)  # [1, seq_len]

        with torch.no_grad():
            _, encoder_hidden = encoder(input_tensor)
            decoder_hidden = encoder_hidden
            decoder_input = torch.tensor([[tgt_tok.char2idx["<sos>"]]])

            generated_indices = []
            for _ in range(max_steps):
                prediction, decoder_hidden = decoder(decoder_input, decoder_hidden)
                top_idx = prediction.argmax(dim=-1).item()

                if top_idx == tgt_tok.char2idx["<eos>"]:
                    break
                if top_idx != tgt_tok.char2idx["<pad>"]:
                    generated_indices.append(top_idx)

                decoder_input = torch.tensor([[top_idx]])

            predicted_word = "".join(tgt_tok.idx2char[idx] for idx in generated_indices)
            results.append((word, predicted_word))

    return results


# --------------------------------------------------------------------------------------
# Sidebar — navigation + status
# --------------------------------------------------------------------------------------

st.sidebar.title("🏹 ScriptMapper")
st.sidebar.caption("Vanilla Seq2Seq · Dakshina Telugu lexicon")

page = st.sidebar.radio(
    "Workspace",
    ["🧪 Transliterate", "📊 Dataset & Tokenizer"],
    label_visibility="collapsed",
)

st.sidebar.divider()

dataset_ready = os.path.exists(TRAIN_FILE)
model_ready = os.path.exists(WEIGHTS_PATH)

st.sidebar.markdown("**Status**")
st.sidebar.markdown(f"{'🟢' if dataset_ready else '🔴'} Dataset file")
st.sidebar.markdown(f"{'🟢' if model_ready else '🔴'} Trained checkpoint")

if not dataset_ready:
    st.sidebar.caption(f"Expected at `{TRAIN_FILE}`")
if not model_ready:
    st.sidebar.caption(f"Expected at `{WEIGHTS_PATH}`")

# --------------------------------------------------------------------------------------
# Page: Transliterate (inference)
# --------------------------------------------------------------------------------------

if page == "🧪 Transliterate":
    st.title("Transliteration Sandbox")
    st.caption("Type a romanized Telugu phrase and let the trained Seq2Seq model map it to native script.")

    encoder, decoder, src_tok, tgt_tok = load_model(WEIGHTS_PATH)

    if encoder is None:
        st.warning(
            f"No trained checkpoint found at `{WEIGHTS_PATH}`. "
            "Train a model first, or place your downloaded `.pth` file in that folder."
        )
    else:
        st.success("Model weights loaded — ready to transliterate.")

        user_input = st.text_input(
            "Romanized text (English letters):",
            value="amma ela unnav",
            help="Multiple words are transliterated independently, then stitched back together.",
        ).strip()

        st.caption("Updates live as you type — no button needed.")

        if user_input:
            word_pairs = transliterate_sentence(user_input, encoder, decoder, src_tok, tgt_tok)
            final_output = " ".join(pred for _, pred in word_pairs)

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Your input**")
                st.markdown(f'<div class="mono-box">{user_input}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown("**Model output**")
                st.markdown(f'<div class="telugu-output">{final_output}</div>', unsafe_allow_html=True)

            if len(word_pairs) > 1:
                with st.expander("Per-word breakdown"):
                    breakdown_df = pd.DataFrame(word_pairs, columns=["Input word", "Predicted script"])
                    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------------
# Page: Dataset & Tokenizer
# --------------------------------------------------------------------------------------

else:
    st.title("Dataset & Tokenizer Inspector")
    st.caption("Explore the raw Dakshina lexicon and how characters get mapped to indices.")

    if not dataset_ready:
        st.error(f"Dataset not found at `{TRAIN_FILE}`. Please ensure your Telugu files are unpacked correctly.")
        st.stop()

    raw_df = load_dataset(TRAIN_FILE)
    src_tokenizer, tgt_tokenizer = build_tokenizers(
        tuple(raw_df["Romanized"].astype(str).tolist()),
        tuple(raw_df["Telugu Word"].astype(str).tolist()),
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📋 Raw data")
        st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)
        st.metric("Total training pairs", f"{len(raw_df):,}")

    with col2:
        st.subheader("🔢 Vocabulary")
        m1, m2 = st.columns(2)
        m1.metric("Source (Latin)", len(src_tokenizer.char2idx))
        m2.metric("Target (Telugu)", len(tgt_tokenizer.char2idx))

        with st.expander("View Telugu character map (char2idx)"):
            st.json(tgt_tokenizer.char2idx)
        with st.expander("View Latin character map (char2idx)"):
            st.json(src_tokenizer.char2idx)

    st.divider()
    st.subheader("🧪 Tokenizer sandbox")
    st.caption("See how a word becomes a tensor, and confirm `decode` reverses it correctly.")

    sandbox_input = st.text_input("Romanized word to test:", "amma", key="tokenizer_sandbox")

    if sandbox_input:
        encoded_tensor = src_tokenizer.encode(sandbox_input.strip())
        decoded_output = src_tokenizer.decode(encoded_tensor)
        roundtrip_ok = decoded_output.strip() == sandbox_input.strip().lower() or decoded_output.strip() == sandbox_input.strip()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Input string**")
            st.markdown(f'<div class="mono-box">{sandbox_input}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown("**Encoded tensor**")
            st.markdown(f'<div class="mono-box">{encoded_tensor.tolist()}</div>', unsafe_allow_html=True)
        with c3:
            st.markdown("**Decoded word** " + ("✅" if roundtrip_ok else "⚠️"))
            st.markdown(f'<div class="mono-box">{decoded_output}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("📦 Mini-batch padding check")
    st.caption("Confirms `collate_fn` pads variable-length words correctly within a batch.")

    try:
        dataset = TransliterationDataset(TRAIN_FILE, src_tokenizer, tgt_tokenizer)
        data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        src_batch, tgt_batch = next(iter(data_loader))

        b1, b2 = st.columns(2)
        b1.metric("Source batch shape", str(list(src_batch.shape)))
        b2.metric("Target batch shape", str(list(tgt_batch.shape)))

        st.caption("Padded source matrix (rows padded with `0` to match the longest word in the batch):")
        st.dataframe(pd.DataFrame(src_batch.numpy()), use_container_width=False)
    except StopIteration:
        st.warning("Dataset produced no batches — check that the TSV file has rows.")