import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline

vectorstore = None


def load_documents(files):
    documents = []

    for file in files:
        path = file.name

        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path)
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue

        documents.extend(loader.load())

    return documents


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_vectorstore(files):
    docs = load_documents(files)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(splits, embeddings)


def build_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text2text-generation",
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

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def chat(files, question):
    global vectorstore

    if not files:
        return "Please upload at least one document."

    if vectorstore is None:
        vectorstore = build_vectorstore(files)

    chain = build_chain(vectorstore)
    return chain.invoke(question)


iface = gr.Interface(
    fn=chat,
    inputs=[
        gr.File(file_types=[".pdf", ".txt", ".docx"], file_count="multiple"),
        gr.Textbox(label="Ask a question")
    ],
    outputs=gr.Textbox(label="Answer", lines=10),
    title="📄 Doc Query RAG"
)

iface.launch()
