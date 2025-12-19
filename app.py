# app.py

import os
import gradio as gr

from transformers import pipeline
from unstructured.partition.docx import partition_docx

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document


# =================================================
# Document Loading & Text Extraction
# =================================================

def extract_text_from_file(path: str) -> str:
    """Extract raw text safely from PDF, DOCX, or TXT."""
    ext = os.path.splitext(path)[1].lower()
    text = ""

    try:
        # -------- PDF --------
        if ext == ".pdf":
            loader = PyPDFLoader(path)
            pages = loader.load()
            text = "\n".join(
                p.page_content for p in pages if p.page_content.strip()
            )

        # -------- DOCX --------
        elif ext in [".docx", ".doc"]:
            elements = partition_docx(path)
            text = "\n".join(
                el.text for el in elements if hasattr(el, "text") and el.text.strip()
            )

        # -------- TXT --------
        elif ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")

    return text.strip()


def load_and_split_docs(filepaths):
    """Convert extracted text into LangChain Documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    documents = []

    for path in filepaths:
        raw_text = extract_text_from_file(path)

        if not raw_text:
            print(f"[WARN] No text found in {path}")
            continue

        doc = Document(
            page_content=raw_text,
            metadata={"source": os.path.basename(path)}
        )

        chunks = splitter.split_documents([doc])
        documents.extend(chunks)

        print(f"[OK] Loaded {len(chunks)} chunks from {path}")

    return documents


# =================================================
# RAG Chain
# =================================================

def build_rag_chain(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    hf_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipe)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )


# =================================================
# Gradio Callback
# =================================================

def answer_question(files, question):
    if not files:
        return "❗ Please upload at least one document."

    if not question.strip():
        return "❗ Please enter a question."

    docs = load_and_split_docs(files)

    if not docs:
        return "❗ No readable text found in uploaded documents."

    try:
        rag_chain = build_rag_chain(docs)
        return rag_chain.run(question)
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# =================================================
# Gradio UI
# =================================================

with gr.Blocks() as demo:
    gr.Markdown("# 📄 Multi-Document RAG Chatbot")
    gr.Markdown("Upload **PDF / DOCX / TXT** files and ask questions about them.")

    doc_files = gr.File(
        label="Upload Documents",
        file_types=[".pdf", ".docx", ".doc", ".txt"],
        file_count="multiple",
        type="filepath"
    )

    question = gr.Textbox(
        label="Your Question",
        placeholder="What is this document about?"
    )

    answer = gr.Textbox(
        label="Answer",
        lines=6
    )

    gr.Button("Get Answer").click(
        fn=answer_question,
        inputs=[doc_files, question],
        outputs=answer
    )


# =================================================
# Run App
# =================================================

if __name__ == "__main__":
    demo.launch()



