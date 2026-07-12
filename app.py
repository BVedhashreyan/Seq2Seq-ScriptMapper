import torch
import streamlit as st
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Nirmala UI",
    "Gautami",
    "Arial Unicode MS",
]

from utils.dataset import CharacterTokenizer
from utils.inference import greedy_decode_vanilla, greedy_decode_attention
from models.vanilla_seq2seq import Encoder as VanillaEncoder, Decoder as VanillaDecoder, VanillaSeq2Seq
from models.attention_seq2seq import Encoder as AttentionEncoder, Decoder as AttentionDecoder, AttentionSeq2Seq


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VANILLA_CHECKPOINT = "predictions_vanilla/best_vanilla_model.pth"
ATTENTION_CHECKPOINT = "predictions_attention/best_attention_model.pth"
MAX_LEN = 50


def restore_tokenizers(checkpoint):
    src_tokenizer = CharacterTokenizer(is_target=False)
    tgt_tokenizer = CharacterTokenizer(is_target=True)

    src_tokenizer.char2idx = checkpoint["src_vocab"]
    tgt_tokenizer.char2idx = checkpoint["tgt_vocab"]

    src_tokenizer.idx2char = {idx: char for char, idx in src_tokenizer.char2idx.items()}
    tgt_tokenizer.idx2char = {idx: char for char, idx in tgt_tokenizer.char2idx.items()}

    return src_tokenizer, tgt_tokenizer


@st.cache_resource
def load_vanilla_model():
    checkpoint = torch.load(VANILLA_CHECKPOINT, map_location=DEVICE)
    config = checkpoint["config"]
    src_tokenizer, tgt_tokenizer = restore_tokenizers(checkpoint)

    encoder = VanillaEncoder(
        len(src_tokenizer.char2idx), config["embedding_dim"], config["hidden_dim"],
        config["cell_type"], config["num_layers"], dropout=0.0
    )

    decoder = VanillaDecoder(
        len(tgt_tokenizer.char2idx), config["embedding_dim"], config["hidden_dim"],
        config["cell_type"], config["num_layers"], dropout=0.0
    )

    model = VanillaSeq2Seq(encoder, decoder).to(DEVICE)
    model.encoder.load_state_dict(checkpoint["encoder_state"])
    model.decoder.load_state_dict(checkpoint["decoder_state"])
    model.eval()

    return model, src_tokenizer, tgt_tokenizer


@st.cache_resource
def load_attention_model():
    checkpoint = torch.load(ATTENTION_CHECKPOINT, map_location=DEVICE)
    config = checkpoint["config"]
    src_tokenizer, tgt_tokenizer = restore_tokenizers(checkpoint)

    encoder = AttentionEncoder(
        len(src_tokenizer.char2idx), config["embedding_dim"], config["hidden_dim"],
        config["cell_type"], config["num_layers"], dropout=0.0
    )

    decoder = AttentionDecoder(
        len(tgt_tokenizer.char2idx), config["embedding_dim"], config["hidden_dim"],
        config["cell_type"], config["attention_type"],
        config["num_layers"], dropout=0.0
    )

    model = AttentionSeq2Seq(encoder, decoder).to(DEVICE)
    model.encoder.load_state_dict(checkpoint["encoder_state"])
    model.decoder.load_state_dict(checkpoint["decoder_state"])
    model.eval()

    return model, src_tokenizer, tgt_tokenizer


def plot_attention(src, predictions, attention, src_tokenizer, tgt_tokenizer):
    source_labels = [
        src_tokenizer.idx2char[token.item()]
        for token in src[0]
        if token.item() != src_tokenizer.char2idx["<pad>"]
    ]

    prediction_labels = []
    eos_idx = tgt_tokenizer.char2idx["<eos>"]

    for token in predictions[0]:
        token = token.item()

        if token == eos_idx:
            prediction_labels.append("<eos>")
            break

        if token not in (
            tgt_tokenizer.char2idx["<pad>"],
            tgt_tokenizer.char2idx["<sos>"]
        ):
            prediction_labels.append(tgt_tokenizer.idx2char[token])

    attention_matrix = attention[
        0,
        :len(prediction_labels),
        :len(source_labels)
    ].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(attention_matrix, aspect="auto")

    ax.set_xticks(range(len(source_labels)))
    ax.set_xticklabels(source_labels)

    ax.set_yticks(range(len(prediction_labels)))
    ax.set_yticklabels(prediction_labels)

    ax.set_xlabel("Romanized Input Characters")
    ax.set_ylabel("Predicted Telugu Characters")
    ax.set_title("Attention Alignment")

    fig.colorbar(image, ax=ax, label="Attention Weight")
    fig.tight_layout()

    return fig


st.set_page_config(page_title="Telugu Transliteration", page_icon="🔤")

st.title("Telugu Translit")
st.write("Character-level transliteration using Vanilla Seq2Seq and Attention Seq2Seq models.")

model_choice = st.selectbox(
    "Select Model",
    ["Vanilla Seq2Seq", "Attention Seq2Seq"]
)

with st.form("transliteration_form"):
    sentence = st.text_input("Enter Romanized Telugu Text", placeholder="Example: amma ela unnav")
    submitted = st.form_submit_button("Transliterate", type="primary")

show_attention = model_choice == "Attention Seq2Seq" and st.checkbox("Show Attention Heatmap", value=True)

if model_choice == "Vanilla Seq2Seq":
    model, src_tokenizer, tgt_tokenizer = load_vanilla_model()
else:
    model, src_tokenizer, tgt_tokenizer = load_attention_model()

if "results" not in st.session_state:
    st.session_state.results = None
if "predicted_sentence" not in st.session_state:
    st.session_state.predicted_sentence = None
if "result_model" not in st.session_state:
    st.session_state.result_model = None

if submitted:
    if not sentence.strip():
        st.warning("Enter Romanized Telugu text.")
    else:
        sos_idx = tgt_tokenizer.char2idx["<sos>"]
        eos_idx = tgt_tokenizer.char2idx["<eos>"]
        results = []

        for word in sentence.split():
            clean_word = "".join(c for c in word if c.isalnum()).lower()
            if not clean_word:
                continue

            src = src_tokenizer.encode(clean_word).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                if model_choice == "Vanilla Seq2Seq":
                    predictions = greedy_decode_vanilla(model, src, sos_idx, eos_idx, MAX_LEN)
                    attention = None
                else:
                    predictions, attention = greedy_decode_attention(model, src, sos_idx, eos_idx, MAX_LEN, return_attention=True)

            results.append({
                "input": clean_word,
                "prediction": tgt_tokenizer.decode(predictions[0]),
                "src": src,
                "predictions": predictions,
                "attention": attention,
            })

        st.session_state.results = results
        st.session_state.predicted_sentence = " ".join(result["prediction"] for result in results)
        st.session_state.result_model = model_choice

if st.session_state.results is not None and st.session_state.result_model == model_choice:
    st.subheader("Prediction")
    st.success(st.session_state.predicted_sentence)

    if show_attention:
        results = st.session_state.results
        selected_index = st.selectbox("Select word for attention heatmap", range(len(results)), format_func=lambda i: f"{i + 1}. {results[i]['input']}")
        selected = results[selected_index]

        fig = plot_attention(selected["src"], selected["predictions"], selected["attention"], src_tokenizer, tgt_tokenizer)
        st.pyplot(fig)
        plt.close(fig)