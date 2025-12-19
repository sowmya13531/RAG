import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline

# ------------------------------
# Config
# ------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3  # Number of retrieved chunks

# ------------------------------
# Chat history
# ------------------------------
chat_history = []

# ------------------------------
# Document Loader (Multi-format)
# ------------------------------
def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext in [".doc", ".docx"]:
        loader = UnstructuredWordDocumentLoader(path)
    elif ext == ".txt":
        loader = TextLoader(path, encoding="utf-8")
    elif ext == ".csv":
        loader = CSVLoader(file_path=path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    docs = loader.load()
    for i, doc in enumerate(docs):
        doc.metadata["source"] = os.path.basename(path)
        doc.metadata["page"] = i + 1
        doc.metadata["type"] = ext
    return docs

def load_and_split_files(filepaths):
    """Load multiple files and split into chunks."""
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for path in filepaths:
        docs = load_document(path)
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)
    return all_docs

# ------------------------------
# Vectorstore
# ------------------------------
def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# ------------------------------
# LLM
# ------------------------------
def load_llm():
    hf_pipe = pipeline("text2text-generation", model=LLM_MODEL, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=hf_pipe)

PROMPT = PromptTemplate.from_template(
"""
You are a document assistant.
Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)

# ------------------------------
# Simple grounding evaluation
# ------------------------------
def evaluate_answer(answer, context):
    context_words = set(context.lower().split())
    answer_words = set(answer.lower().split())
    overlap = context_words & answer_words
    score = len(overlap) / max(len(answer_words), 1)
    if score > 0.1:
        return "Answer seems grounded"
    else:
        return "Answer may not be fully grounded"

# ------------------------------
# Chatbot QA function
# ------------------------------
def chatbot_answer(files, user_question):
    global chat_history
    
    if not files or not user_question:
        return "Upload documents and ask a question."

    # Load and split files
    docs = load_and_split_files(files)

    # Build vectorstore and retrieve top chunks
    vectorstore = build_vectorstore(docs)
    retrieved_docs = vectorstore.similarity_search(user_question, k=TOP_K)
    retrieved_context = "\n\n".join([d.page_content for d in retrieved_docs])

    # Include chat history in context
    history_text = "\n".join([f"User: {h['question']}\nBot: {h['answer']}" for h in chat_history])
    if history_text:
        full_context = history_text + "\n\n" + retrieved_context
    else:
        full_context = retrieved_context

    # Generate answer
    llm = load_llm()
    parser = StrOutputParser()
    chain = PROMPT | llm | parser
    answer = chain.invoke({"context": full_context, "question": user_question})

    # Save to chat history
    chat_history.append({"question": user_question, "answer": answer})

    # Source citations
    sources = ", ".join([f"{d.metadata['source']} (page {d.metadata['page']})" for d in retrieved_docs])

    # Evaluation
    evaluation = evaluate_answer(answer, retrieved_context)

    final_answer = f"{answer}\n\n📌 Sources: {sources}\n📊 Evaluation: {evaluation}"
    return final_answer

# ------------------------------
# Gradio UI
# ------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Multi-Document Chatbot QA")
    gr.Markdown("Upload PDFs, DOCX, TXT, CSV files and ask questions. Chatbot remembers previous questions!")

    file_input = gr.File(label="Upload Documents", file_types=[".pdf", ".docx", ".txt", ".csv"], type="filepath", file_count="multiple")
    question_input = gr.Textbox(label="Ask a question")
    chat_output = gr.Textbox(label="Chatbot Answer", lines=10)

    gr.Button("Send").click(chatbot_answer, inputs=[file_input, question_input], outputs=[chat_output])

# ------------------------------
# Launch
# ------------------------------
if __name__ == "__main__":
    demo.launch()
