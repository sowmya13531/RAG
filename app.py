import gradio as gr
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# ------------------------------
# Helper functions
# ------------------------------

def load_and_split_docs(filepaths):
    """Load PDFs, Word docs, or TXT files and split into chunks."""
    all_docs = []
    for path in filepaths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".docx") or path.endswith(".doc"):
            loader = UnstructuredWordDocumentLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        all_docs.extend(chunks)
    return all_docs

def build_rag_chain(docs):
    """Create vectorstore retriever and LLM chain."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return rag_chain

def answer_question(files, question):
    """Process documents and answer a question."""
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
    gr.Markdown("Upload PDFs, Word documents, or TXT files and ask questions about them.")

    with gr.Row():
        doc_files = gr.File(
            label="Upload Documents",
            file_types=[".pdf", ".docx", ".doc", ".txt"],
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

