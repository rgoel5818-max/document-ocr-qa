import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import torch


st.set_page_config(page_title="Document OCR + QA", layout="wide")

# ---------------- Load Models ----------------

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    qa = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        device=0 if torch.cuda.is_available() else -1
    )

    return embedder, qa


embedder, qa_pipeline = load_models()

# ---------------- OCR ----------------

def ocr_image(img):
    return pytesseract.image_to_string(img)

def ocr_pdf(pdf_bytes):
    pages = convert_from_bytes(
    pdf_bytes)

    text = ""
    for p in pages:
        text += pytesseract.image_to_string(p)
    return text

# ---------------- Text Chunking ----------------

def chunk_text(text, chunk_size=350, overlap=50):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks

# ---------------- Vector Store ----------------

def build_faiss(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index

def retrieve(question, chunks, index, k=5):
    q_emb = embedder.encode([question])
    D, I = index.search(np.array(q_emb), k)
    return [chunks[i] for i in I[0]]

# ---------------- QA ----------------

def answer_question(context, question):

    result = qa_pipeline(
        question=question,
        context=context
    )

    return result["answer"]


# ---------------- UI ----------------

st.title("📄 Document OCR + Question Answering (Free Models)")

uploaded = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded:

    if uploaded.type == "application/pdf":
        raw_text = ocr_pdf(uploaded.read())
    else:
        img = Image.open(uploaded)
        raw_text = ocr_image(img)

    st.subheader("Extracted Text Preview")
    st.text_area("", raw_text[:3000], height=200)

    with st.spinner("Building knowledge base..."):
        chunks = chunk_text(raw_text)
        index = build_faiss(chunks)

    st.success("Document indexed successfully!")

    question = st.text_input("Ask a question about your document")

    if question:

        with st.spinner("Thinking..."):
            docs = retrieve(question, chunks, index)
            context = "\n".join(docs)
            answer = answer_question(context, question)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            st.write(context)
