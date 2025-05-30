# Retrieval-Augmented Generation (RAG) System using Hugging Face & FAISS

This project implements a Retrieval-Augmented Generation (RAG) system that combines document retrieval with large language model (LLM) generation. It supports multiple data sources such as PDFs and websites, and uses FAISS for fast similarity search and a quantized Zephyr 7B model for efficient response generation.

Project Structure:
rag/
│
├── data/
│   ├── deccanqueen.pdf
│   └── pune.pdf
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── rag4.py                 # Main RAG pipeline
├── requirements.txt        # Python dependencies
└── README.txt              # Project documentation

Features:
- Load documents from PDFs (and optionally from websites, markdown folders, CSVs)
- Split documents into manageable chunks
- Generate embeddings using all-MiniLM-L6-v2
- Vector indexing using FAISS
- Load & run Zephyr 7B LLM using 4-bit quantization
- Efficient GPU usage and memory management
- Interactive query interface

Tech Stack:
- LangChain
- Hugging Face Transformers
- FAISS
- BitsAndBytes (bnb)

Setup Instructions:

1. Clone the Repository
   git clone https://github.com/yourusername/rag
   cd rag

2. Install Dependencies
   pip install -r requirements.txt

3. Set Hugging Face Cache Path (Optional)
   export HF_HOME=D:/huggingface_models/huggingface

4. Download Models
   The required models will be automatically downloaded on the first run and cached in HF_HOME.

How to Run:
python rag4.py

It will:
- Load documents from the data/ folder
- Create or load a FAISS index
- Start an interactive chat loop

Configuration Options:
- embedding_model: Change embedding model (e.g., all-MiniLM-L6-v2)
- llm_model: Hugging Face model name (e.g., HuggingFaceH4/zephyr-7b-alpha)
- chunk_size: Number of characters in each text chunk
- index_path: Path to save/load FAISS index
- data_sources: Add/remove sources (pdf, web, csv, markdown folder)

Cleanup:
At the end of the session, GPU memory is cleared using:
torch.cuda.empty_cache()

License:
This project is licensed under the MIT License.
