"""
Text Summarization App
Summarizes long text, .txt or .pdf files using BART models (abstractive
summarization). Long documents are split into chunks, summarized, and merged.
Runs locally. No API key needed.
"""
import streamlit as st
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# =========================================================
# CUSTOM CSS — CHERRY & BURGUNDY THEME
# =========================================================

st.markdown("""
<style>

/* =========================================================
   GLOBAL APP
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 8% 8%,
            rgba(122, 24, 52, 0.10),
            transparent 28%
        ),
        radial-gradient(
            circle at 92% 12%,
            rgba(190, 72, 96, 0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(246, 211, 218, 0.45),
            transparent 35%
        ),
        #fffaf7;

    color: #3f252d;
}


/* =========================================================
   MAIN CONTAINER
   ========================================================= */

.block-container {
    max-width: 1150px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}


/* =========================================================
   HEADINGS
   ========================================================= */

h1 {
    font-family: Georgia, "Times New Roman", serif !important;
    font-size: 3.2rem !important;
    font-weight: 700 !important;
    text-align: center;

    color: #681d35 !important;

    letter-spacing: -1px;
    margin-bottom: 0.3rem !important;
}

h2, h3 {
    font-family: Georgia, "Times New Roman", serif !important;
    color: #681d35 !important;
    font-weight: 700 !important;
}

p {
    color: #674850;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #f8e9ec 0%,
            #fceff0 50%,
            #f5e2e6 100%
        );

    border-right: 1px solid rgba(104, 29, 53, 0.12);
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #681d35 !important;
}

section[data-testid="stSidebar"] label {
    color: #633d48 !important;
    font-weight: 600;
}


/* =========================================================
   RADIO BUTTONS
   ========================================================= */

div[role="radiogroup"] {
    gap: 8px;
}

div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.65);
    border: 1px solid rgba(128, 48, 69, 0.15);
    border-radius: 12px;
    padding: 8px 14px;

    transition: all 0.2s ease;
}

div[role="radiogroup"] label:hover {
    background: #f8e2e7;
    border-color: #a84b63;
}


/* =========================================================
   TEXT AREA
   ========================================================= */

textarea {
    background: rgba(255, 253, 251, 0.9) !important;

    color: #3f252d !important;

    border: 1px solid #dfbfc8 !important;
    border-radius: 18px !important;

    padding: 18px !important;

    font-size: 15px !important;
    line-height: 1.7 !important;

    box-shadow:
        0 8px 25px rgba(104, 29, 53, 0.06);
}

textarea:focus {
    border-color: #9c4059 !important;

    box-shadow:
        0 0 0 2px rgba(156, 64, 89, 0.12),
        0 8px 25px rgba(104, 29, 53, 0.08);
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    background: rgba(255, 250, 248, 0.75);

    border: 1.5px dashed #b96b7e;
    border-radius: 20px;

    padding: 14px;

    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    background: #fff1f3;

    border-color: #8b2946;

    box-shadow:
        0 10px 30px rgba(104, 29, 53, 0.10);
}


/* =========================================================
   PRIMARY BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;

    border: none !important;
    border-radius: 15px;

    padding: 0.8rem 1.2rem;

    font-size: 16px;
    font-weight: 700;

    color: white !important;

    background:
        linear-gradient(
            135deg,
            #681d35,
            #852843,
            #a33e59
        ) !important;

    box-shadow:
        0 8px 22px rgba(104, 29, 53, 0.25);

    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 12px 30px rgba(104, 29, 53, 0.35);

    background:
        linear-gradient(
            135deg,
            #57172c,
            #76223c,
            #96364f
        ) !important;
}

.stButton > button:active {
    transform: scale(0.98);
}


/* =========================================================
   DOWNLOAD BUTTON
   ========================================================= */

.stDownloadButton > button {
    width: 100%;

    border-radius: 13px !important;

    border: 1px solid #c98b9a !important;

    background: #f9e9ed !important;

    color: #681d35 !important;

    font-weight: 600 !important;
}

.stDownloadButton > button:hover {
    background: #f3dbe1 !important;
    border-color: #9c4059 !important;
}


/* =========================================================
   ORIGINAL + SUMMARY CARDS
   ========================================================= */

div[data-testid="column"] {
    background:
        rgba(255, 253, 251, 0.82);

    border: 1px solid rgba(126, 48, 69, 0.13);

    border-radius: 22px;

    padding: 22px;

    box-shadow:
        0 10px 35px rgba(104, 29, 53, 0.07);
}


/* =========================================================
   SUMMARY RESULT
   ========================================================= */

div[data-testid="stAlert"] {
    background:
        linear-gradient(
            135deg,
            #fae8ec,
            #fff1f1
        ) !important;

    border: 1px solid #d9aeba !important;

    border-radius: 18px !important;

    color: #4d2934 !important;

    padding: 20px !important;

    box-shadow:
        0 8px 25px rgba(104, 29, 53, 0.07);
}

div[data-testid="stAlert"] p {
    color: #54343d !important;
}


/* =========================================================
   EXPANDER
   ========================================================= */

div[data-testid="stExpander"] {
    background: rgba(255, 250, 248, 0.7);

    border: 1px solid #dfc1c9;

    border-radius: 16px;
}

div[data-testid="stExpander"] summary {
    color: #681d35;

    font-weight: 600;
}


/* =========================================================
   SELECTBOX
   ========================================================= */

div[data-baseweb="select"] > div {
    background: rgba(255, 250, 248, 0.8);

    border: 1px solid #dfc1c9;

    border-radius: 12px;
}

div[data-baseweb="select"] > div:hover {
    border-color: #9c4059;
}


/* =========================================================
   PROGRESS BAR
   ========================================================= */

div[data-testid="stProgress"] > div {
    background: #f0d8de;

    border-radius: 20px;

    height: 9px;
}

div[data-testid="stProgress"] > div > div {
    background:
        linear-gradient(
            90deg,
            #681d35,
            #8f304d,
            #bd6076
        );

    border-radius: 20px;
}


/* =========================================================
   CAPTIONS
   ========================================================= */

.stCaption {
    color: #8a626d !important;
}


/* =========================================================
   SPINNER
   ========================================================= */

div[data-testid="stSpinner"] {
    color: #8b2946;
}


/* =========================================================
   CHERRY DECORATIONS
   ========================================================= */

.stApp::before {
    content: "🍒";

    position: fixed;

    top: 75px;
    left: 4%;

    font-size: 25px;

    opacity: 0.45;

    pointer-events: none;

    filter: saturate(0.8);
}

.stApp::after {
    content: "🍒";

    position: fixed;

    bottom: 65px;
    right: 5%;

    font-size: 30px;

    opacity: 0.28;

    pointer-events: none;

    transform: rotate(-12deg);
}


/* =========================================================
   SUBTLE CHERRY ACCENT
   ========================================================= */

[data-testid="stHeader"] {
    background: transparent;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 2rem;
    }

    h1 {
        font-size: 2.3rem !important;
    }

    div[data-testid="column"] {
        padding: 16px;
        margin-bottom: 15px;
    }

}

</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="wide")

MODELS = {
    "DistilBART (faster, lighter)": "sshleifer/distilbart-cnn-12-6",
    "BART Large (better quality)": "facebook/bart-large-cnn",
}

LENGTHS = {  # (min_tokens, max_tokens) per chunk
    "Short": (30, 80),
    "Medium": (60, 150),
    "Long": (100, 250),
}

CHUNK_WORDS = 600  # BART accepts ~1024 tokens; 600 words keeps us safely under


@st.cache_resource(show_spinner="Loading model (first run downloads it)...")
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model


def chunk_text(text: str, size: int = CHUNK_WORDS):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def summarize_chunk(text, tokenizer, model, min_len, max_len):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    ids = model.generate(
        **inputs,
        min_length=min_len,
        max_length=max_len,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def summarize(text, tokenizer, model, min_len, max_len):
    chunks = chunk_text(text)
    progress = st.progress(0.0, text="Summarizing...")
    parts = []
    for i, chunk in enumerate(chunks, 1):
        # Very short chunks can't hit min_len; relax it
        words = len(chunk.split())
        parts.append(summarize_chunk(chunk, tokenizer, model, min(min_len, words // 2), max_len))
        progress.progress(i / len(chunks), text=f"Summarized chunk {i}/{len(chunks)}")
    progress.empty()

    combined = " ".join(parts)
    # If the merged summary is still long, summarize it once more
    if len(chunks) > 1 and len(combined.split()) > CHUNK_WORDS:
        combined = summarize_chunk(combined, tokenizer, model, min_len, max_len * 2)
    return combined


def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
length = st.sidebar.radio("Summary length", list(LENGTHS.keys()), index=1)

# ---------------- Main ----------------
st.title("🍒 Text Summarizer")
st.write("Turn long articles, notes, and documents into clear, concise summaries.")
st.write("Paste text or upload a file (.txt / .pdf) to get a concise summary.")

source = st.radio("Input type", ["Paste text", "Upload file"], horizontal=True)
text = ""
if source == "Paste text":
    text = st.text_area("Your text", height=300, placeholder="Paste an article, notes, or report here...")
else:
    file = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])
    if file:
        text = read_pdf(file) if file.name.lower().endswith(".pdf") else file.read().decode("utf-8", errors="ignore")
        with st.expander("Preview extracted text"):
            st.write(text[:3000] + ("..." if len(text) > 3000 else ""))

if st.button("⚡ Summarize", type="primary"):
    word_count = len(text.split())
    if word_count < 50:
        st.warning("Please provide at least 50 words of text.")
    else:
        tokenizer, model = load_model(MODELS[model_name])
        min_len, max_len = LENGTHS[length]
        summary = summarize(text, tokenizer, model, min_len, max_len)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.caption(f"{word_count} words")
            st.write(text[:5000] + ("..." if len(text) > 5000 else ""))
        with col2:
            st.subheader("Summary")
            s_count = len(summary.split())
            st.caption(f"{s_count} words • {100 - s_count * 100 // word_count}% shorter")
            st.success(summary)
            st.download_button("⬇️ Download summary", summary, "summary.txt", "text/plain")