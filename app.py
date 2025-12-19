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

from transformers import pipeline


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
    retriever = vectorstore.as_retriever()

    llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256
        )
    )

    prompt = PromptTemplate.from_template(
        """Use the following context to answer the question.

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = (
        {
            "context": retriever,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def chat(files, question):
    chain = build_chain(files)
    return chain.invoke(question)


iface = gr.Interface(
    fn=chat,
    inputs=[
        gr.File(file_types=[".pdf", ".txt", ".docx"], file_count="multiple"),
        gr.Textbox(label="Ask a question")
    ],
    outputs=gr.Textbox(
        label = 'Answer',
        lines=20,
        max_lines=40,
        interactive=False
    ),
    title="Doc Query RAG"
)

iface.launch()