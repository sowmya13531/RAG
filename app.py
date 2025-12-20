import gradio as gr

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from transformers import pipeline


# ---------------------------
# Load documents
# ---------------------------
def load_documents(files):
    documents = []
    for file in files:
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file.name)
        elif file.name.endswith(".txt"):
            loader = TextLoader(file.name)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file.name)
        else:
            continue

        documents.extend(loader.load())

    return documents


# ---------------------------
# Format documents correctly
# ---------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------
# Build RAG chain
# ---------------------------
def build_chain(files):
    docs = load_documents(files)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256
        )
    )

    prompt = PromptTemplate.from_template(
        """Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."
Context:
{context}
Question:
{question}
Answer:
"""
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ---------------------------
# Chat function
# ---------------------------
def chat(files, question):
    if not files:
        return "Please upload at least one document."

    chain = build_chain(files)
    return chain.invoke(question)


# ---------------------------
# Gradio UI
# ---------------------------
iface = gr.Interface(
    fn=chat,
    inputs=[
        gr.File(
            file_types=[".pdf", ".txt", ".docx"],
            file_count="multiple",
            label="Upload documents"
        ),
        gr.Textbox(
            label="Ask a question",
            placeholder="What is this document about?"
        )
    ],
    outputs=gr.Textbox(
        label="Answer",
        lines=15,
        max_lines=30,
        interactive=False
    ),
    title="📄 Doc Query RAG",
    description="Upload documents(PDF, DOCX, TXT) and ask questions based only on their content."
)

iface.launch()
