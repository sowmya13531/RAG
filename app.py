import gradio as gr
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os

# ------------------------------
# Document Loading Helper
# ------------------------------
def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext in [".txt"]:
        loader = TextLoader(path)
    elif ext in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(path)
    else:
        return []  # unsupported file
    return loader.load()

def load_and_split_files(filepaths):
    all_docs = []
    for path in filepaths:
        docs = load_document(path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)
    return all_docs

# ------------------------------
# Build RAG Chatbot
# ------------------------------
def build_rag_chatbot(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})

    hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False
    )
    return qa_chain

# ------------------------------
# Gradio Function
# ------------------------------
chatbot_chain = None

def chat_with_docs(files, question):
    global chatbot_chain
    if chatbot_chain is None:
        docs = load_and_split_files(files)
        if not docs:
            return "No supported documents found."
        chatbot_chain = build_rag_chatbot(docs)
    return chatbot_chain.run(question)

# ------------------------------
# Gradio UI
# ------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Multi-Document Chatbot")
    gr.Markdown("Upload any combination of PDF, Word, or TXT files and ask questions about them.")

    files_input = gr.File(file_types=[".pdf", ".txt", ".docx", ".doc"], type="filepath", file_count="multiple")
    question_input = gr.Textbox(label="Ask a question")
    answer_output = gr.Textbox(label="Answer")

    submit_btn = gr.Button("Ask")
    submit_btn.click(chat_with_docs, inputs=[files_input, question_input], outputs=[answer_output])

if __name__ == "__main__":
    demo.launch()

