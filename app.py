import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.memory import ConversationBufferMemory
from transformers import pipeline
from operator import itemgetter

# ------------------------------
# Global memory for conversation
# ------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat_history = []

# ------------------------------
# Helper functions
# ------------------------------

def load_and_split_files(filepaths):
    """Load PDF, DOCX, and TXT files and split them into chunks."""
    all_docs = []
    for path in filepaths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue  # skip unsupported files

        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        all_docs.extend(chunks)
    return all_docs

def build_rag_chain_with_memory(docs, memory):
    """Create vectorstore, retriever, LLM, prompt, and RAG chain with conversational memory."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt = PromptTemplate.from_template(
        """You are a helpful assistant. Use the following conversation and context to answer the question.
Conversation history:
{chat_history}
Context:
{context}
Question:
{question}
"""
    )

    rag_chain = (
        {
            "context": retriever,
            "question": itemgetter("question"),
            "chat_history": memory.buffer
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def chat_rag(files, question):
    """Main function to process files, maintain memory, and answer questions."""
    if not files or not question:
        return "Upload documents and ask a question.", chat_history

    docs = load_and_split_files(files)
    if not docs:
        return "No valid documents found.", chat_history

    rag_chain = build_rag_chain_with_memory(docs, memory)
    answer = rag_chain.invoke({"question": question})

    chat_history.append((question, answer))
    return answer, chat_history

# ------------------------------
# Gradio UI
# ------------------------------

with gr.Blocks() as demo:
    gr.Markdown("# 📄 Chat-based PDF, DOCX, TXT RAG")
    gr.Markdown("Upload PDF, DOCX, or TXT files and ask questions about them in a chat interface.")

    with gr.Row():
        files_input = gr.File(
            label="Upload Docs",
            file_types=[".pdf", ".docx", ".txt"],
            type="filepath",
            file_count="multiple"
        )
    question_input = gr.Textbox(label="Ask a question")
    answer_output = gr.Textbox(label="Answer")
    chat_output = gr.Chatbot(label="Chat History")

    submit_btn = gr.Button("Send")
    submit_btn.click(chat_rag, inputs=[files_input, question_input], outputs=[answer_output, chat_output])

# ------------------------------
# Run locally
# ------------------------------
if __name__ == "__main__":
    demo.launch()



