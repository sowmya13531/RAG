# app.py

import gradio as gr
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document

from transformers import pipeline
from unstructured.partition.docx import partition_docx

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def load_and_split_docs(filepaths):
    all_docs = []

    for path in filepaths:
        ext = os.path.splitext(path)[1].lower()

        try:
            # ---------- PDF ----------
            if ext == ".pdf":
                loader = PyPDFLoader(path)
                docs = loader.load()

            # ---------- DOCX ----------
            elif ext in [".docx", ".doc"]:
                elements = partition_docx(path)
                text = "\n".join(str(el) for el in elements)
                docs = [Document(
                    page_content=text,
                    metadata={"source": os.path.basename(path)}
                )]

            # ---------- TXT ----------
            elif ext == ".txt":
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()

            else:
                continue

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(docs)
            all_docs.extend(chunks)

        except Exception as e:
            print(f"Error processing {path}: {e}")

    return all_docs


def build_rag_chain(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    return rag_chain


def answer_question(files, question):
    if not files:
        return "❗ Please upload at least one document."

    if not question.strip():
        return "❗ Please enter a question."

    docs = load_and_split_docs(files)

    if len(docs) == 0:
        return "❗ No readable text found in uploaded documents."

    try:
        rag_chain = build_rag_chain(docs)
        return rag_chain.run(question)
    except Exception as e:
        return f"⚠️ Error while answering: {str(e)}"


# -------------------------------------------------
# Gradio UI
# -------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("# 📄 Multi-Document RAG Chatbot")
    gr.Markdown("Upload multiple PDFs, Word documents, or TXT files and ask questions.")

    with gr.Row():
        doc_files = gr.File(
            label="Upload Documents",
            file_types=[".pdf", ".txt", ".docx", ".doc"],
            file_count="multiple",
            type="filepath"
        )

    question_input = gr.Textbox(
        label="Ask a question",
        placeholder="What is the document about?"
    )

    answer_output = gr.Textbox(label="Answer", lines=6)

    submit_btn = gr.Button("Get Answer")

    submit_btn.click(
        fn=answer_question,
        inputs=[doc_files, question_input],
        outputs=answer_output
    )

# -------------------------------------------------
# Run App
# -------------------------------------------------

if __name__ == "__main__":
    demo.launch()




