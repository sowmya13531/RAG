# app.py

import os
import gradio as gr

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredHTMLLoader
)

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & Vectorstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM wrapper
from langchain_community.llms import HuggingFacePipeline

# Correct import for LangChain 0.1.x
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

# Conversation memory
from langchain.memory import ConversationBufferMemory

# HuggingFace pipeline
from transformers import pipeline

# ---------------------------------------
# Global QA chain
# ---------------------------------------
qa_chain = None

# ---------------------------------------
# Load and split documents
# ---------------------------------------
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

# ---------------------------------------
# Build Conversational RAG chain
# ---------------------------------------
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

# ---------------------------------------
# Convert LangChain memory to Gradio tuples
# ---------------------------------------
def get_chat_history_tuples(memory_messages):
    chat_history = []
    # Messages stored in order: user -> assistant -> user -> assistant ...
    for i in range(0, len(memory_messages), 2):
        user_msg = memory_messages[i]["content"] if i < len(memory_messages) else ""
        ai_msg = memory_messages[i+1]["content"] if i+1 < len(memory_messages) else ""
        chat_history.append((user_msg, ai_msg))
    return chat_history

# ---------------------------------------
# Chat handler
# ---------------------------------------
def chat_with_docs(files, user_message, chat_history):
    global qa_chain

    if not files:
        chat_history.append(("System", "❌ Please upload documents first."))
        return chat_history, ""

    # Initialize QA chain if not yet done
    if qa_chain is None:
        docs = load_documents(files)
        qa_chain = build_conversational_rag(docs)

    # Ask the question
    result = qa_chain({"question": user_message})

    # Convert memory to Gradio-compatible format
    chat_history = get_chat_history_tuples(qa_chain.memory.chat_memory.messages)

    return chat_history, ""

# ---------------------------------------
# Gradio UI
# ---------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Conversational Multi-Document RAG Chatbot")
    gr.Markdown(
        "Upload documents (PDF, DOCX, TXT, CSV, HTML) and chat with them using AI."
    )

    file_upload = gr.File(
        label="Upload Documents",
        file_types=[".pdf", ".docx", ".txt", ".csv", ".html"],
        file_count="multiple",
        type="filepath"
    )

    chatbot = gr.Chatbot()
    user_input = gr.Textbox(
        placeholder="Ask a question about your documents..."
    )
    send_btn = gr.Button("Send")

    send_btn.click(
        chat_with_docs,
        inputs=[file_upload, user_input, chatbot],
        outputs=[chatbot, user_input]
    )

# ---------------------------------------
# Run app
# ---------------------------------------
if __name__ == "__main__":
    demo.launch()



