# app.py

import os
import gradio as gr

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredHTMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

# ✅ FIXED IMPORT
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from langchain.memory import ConversationBufferMemory
from transformers import pipeline

qa_chain = None

def load_documents(filepaths):
    all_docs = []

    for path in filepaths:
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext == ".docx":
            loader = Docx2txtLoader(path)
        elif ext == ".txt":
            loader = TextLoader(path)
        elif ext == ".csv":
            loader = CSVLoader(path)
        elif ext == ".html":
            loader = UnstructuredHTMLLoader(path)
        else:
            continue

        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(all_docs)

def build_conversational_rag(docs):
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

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

def chat_with_docs(files, user_message, chat_history):
    global qa_chain

    if not files:
        chat_history.append(("System", "Please upload documents first."))
        return chat_history, ""

    if qa_chain is None:
        docs = load_documents(files)
        qa_chain = build_conversational_rag(docs)

    result = qa_chain({"question": user_message})
    chat_history.append((user_message, result["answer"]))

    return chat_history, ""

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Conversational Multi-Document RAG Chatbot")

    file_upload = gr.File(
        label="Upload Documents",
        file_types=[".pdf", ".docx", ".txt", ".csv", ".html"],
        file_count="multiple",
        type="filepath"
    )

    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Ask about your documents...")
    send_btn = gr.Button("Send")

    send_btn.click(
        chat_with_docs,
        inputs=[file_upload, user_input, chatbot],
        outputs=[chatbot, user_input]
    )

if __name__ == "__main__":
    demo.launch()




