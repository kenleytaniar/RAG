import streamlit as st
import os
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pypdf import PdfReader
import faiss

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP = 50


def load_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def split_into_chunks(text: str, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner=False)
def load_llm():
    tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(LLM_MODEL)
    return tokenizer, model

def generate_answer(tokenizer, model, context, question):
    prompt = f"summarize: Context: {context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


st.title("PDF RAG Chatbot")
st.write("Upload PDF → Extract → Embed → Ask Questions")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    # Load text
    with st.spinner("Reading PDF..."):
        text = load_text_from_pdf(uploaded_file)

    st.success("PDF loaded!")
    st.write(f"Document length: **{len(text)} characters**")

    # Chunking
    with st.spinner("Chunking document..."):
        chunks = split_into_chunks(text)
    st.write(f"Created **{len(chunks)} chunks**")

    # Embeddings
    embedder = load_embedder()
    with st.spinner("Building embeddings..."):
        vectors = embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)

    # Build FAISS
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    tokenizer, model = load_llm()
    st.success("RAG system ready! Ask questions below.")

    question = st.text_input("Enter your question:")
    if st.button("Ask"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            # Search
            q_vec = embedder.encode([question], convert_to_numpy=True)
            _, I = index.search(q_vec, k=3)

            retrieved_chunks = [chunks[i] for i in I[0]]
            context = "\n\n".join(retrieved_chunks)

            st.subheader("📄 Retrieved Chunks")
            for c in retrieved_chunks:
                st.write(f"- {c[:200]}...")

            # Generate answer
            with st.spinner("Generating answer..."):
                answer = generate_answer(tokenizer, model, context, question)

            st.subheader("💡 Answer")
            st.write(answer)
