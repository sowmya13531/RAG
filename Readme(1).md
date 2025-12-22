# 📄 Document Question Answering (RAG) App
A **Retrieval-Augmented Generation (RAG)** application that allows users to upload documents and ask questions. The system retrieves relevant content from the uploaded files and generates **accurate answers strictly grounded in the document context**.

🚀 **Live Demo (Hugging Face Space):**
👉 [HuggingFace App](https://huggingface.co/spaces/Sowmya135/RetrievalAugmentedGenerator)

## 🚀 Features
* 📁 Upload multiple documents (`PDF`, `TXT`, `DOCX`)
* 🔍 Semantic search using **FAISS vector database**
* 🧠 Context-aware answers using **FLAN-T5**
* ❌ Prevents hallucinations (answers only from documents)
* 🌐 Clean web UI built with **Gradio**
* 🔐 No paid APIs or token billing

## 🛠️ Tech Stack
* **Python**
* **Gradio** – User Interface
* **LangChain** – RAG pipeline
* **Hugging Face Transformers**
* **Sentence Transformers** – Text Embeddings
* **FAISS** – Vector Similarity Search

## 🧠 How It Works (RAG Flow)
1. User uploads one or more documents
2. Documents are loaded and split into overlapping chunks
3. Each chunk is converted into vector embeddings
4. Embeddings are stored in a FAISS vector database
5. User asks a question
6. Relevant chunks are retrieved via semantic similarity
7. The LLM generates an answer **only from the retrieved context**

## 📂 Supported File Types
* `.pdf`
* `.txt`
* `.docx`

## ⚙️ Model Details

### 🔹 Embedding Model

* `sentence-transformers/all-MiniLM-L6-v2`

### 🔹 Language Model

* `google/flan-t5-base`

> This is an **open-source LLM** running locally inside Hugging Face Spaces.

## 💰 Cost Clarification (Important)

✅ **No token-based billing**
✅ **No API keys required**
✅ **No OpenAI / paid APIs used**

The model runs **locally on Hugging Face Spaces**, so users can freely interact with the app without incurring costs.

## 📥 Clone the Repository from HuggingFace Spaces

```bash
git clone https://huggingface.co/spaces/Sowmya135/RetrievalAugmentedGenerator
cd RetrievalAugmentedGenerator
```

## 📥 Clone the Repository from Github

```bash
git clone https://github.com/Sowmya135/RAG
cd RAG
```

## ▶️ Run Locally (Step-by-Step)

### 1️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install gradio langchain langchain-community langchain-huggingface \
transformers sentence-transformers faiss-cpu pypdf docx2txt
```

### 3️⃣ Run the Application

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:7860
```

## ⚠️ NOTE
* Vector store is built once per session
* Restart the Space to upload new documents
* Large documents may take longer to process
* Runs on CPU by default (GPU improves speed if enabled)

## 📸 Example Use Cases
* Academic document Q&A
* Research paper exploration
* Resume or report analysis
* Study material querying

## ⭐ Final Note

This project demonstrates **real-world RAG architecture** using only **open-source tools**, making it ideal for learning, showcasing, and deployment without cost concerns.
