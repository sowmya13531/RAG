# app.py
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import os

# ------------------------------
# Helper functions
# ------------------------------

def load_and_split_docs(filepaths):
    """Load multiple files (PDF, DOCX, TXT) and split into chunks."""
    from unstructured.partition.docx import partition_docx

    all_docs = []
    for path in filepaths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)
            docs = loader.load()
        elif ext in [".docx", ".doc"]:
            docs = partition_docx(path)
            docs = [{"page_content": str(doc)} for doc in docs]
        elif ext == ".txt":
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
        else:
            continue

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)
    return all_docs

def build_rag_chain(docs):
    """Create vectorstore, retriever, LLM, and RAG chain."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return rag_chain

def answer_question(files, question):
    """Main function to process documents and answer a question."""
    if not files or not question:
        return "Please upload documents and enter a question."

    docs = load_and_split_docs(files)
    rag_chain = build_rag_chain(docs)
    response = rag_chain.run(question)
    return response

# ------------------------------
# Gradio UI
# ------------------------------

with gr.Blocks() as demo:
    gr.Markdown("# 📄 Multi-Document RAG Chatbot")
    gr.Markdown("Upload multiple PDFs, Word documents, or TXT files and ask questions about them.")

    with gr.Row():
        doc_files = gr.File(
            label="Upload Documents",
            file_types=[".pdf", ".txt", ".docx", ".doc"],
            type="filepath",
            file_count="multiple"
        )
    question_input = gr.Textbox(label="Ask a question about the documents")
    answer_output = gr.Textbox(label="Answer")

    submit_btn = gr.Button("Get Answer")
    submit_btn.click(answer_question, inputs=[doc_files, question_input], outputs=[answer_output])

# ------------------------------
# Run locally
# ------------------------------
if __name__ == "__main__":
    demo.launch()

