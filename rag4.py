from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import BitsAndBytesConfig
import torch
import os

class RAGSystem:
    def __init__(self, config):
        self.config = config
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config["embedding_model"]
        )
        
        # Set HuggingFace cache directory
        os.environ["HF_HOME"] = config.get("hf_home", "D:/huggingface_models/huggingface")
        
        # Initialize components
        self._init_llm()
        self._init_vectorstore()
    
    def _init_llm(self):
        """Initialize the LLM only once"""
        print("Initializing LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["llm_model"],
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config["llm_model"],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            **self.config["generation_config"]
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        print("LLM initialized successfully!")
    
    def _init_vectorstore(self):
        """Initialize vectorstore only once"""
        print("Initializing vectorstore...")
        if self.config["load_existing_index"]:
            self.vectorstore = FAISS.load_local(
                self.config["index_path"],
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            self.vectorstore = self._create_vectorstore()
        print("Vectorstore ready!")
    
    def _create_vectorstore(self):
        """Multi-source document loader"""
        documents = []
        
        for source in self.config["data_sources"]:
            try:
                loader = self._get_loader(source)
                loaded_docs = loader.load()
                
                # Add source metadata
                for doc in loaded_docs:
                    doc.metadata["source"] = source["path"]
                
                documents.extend(loaded_docs)
            except Exception as e:
                print(f"Failed to load {source['path']}: {str(e)}")
                continue
        
        # Split and index
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"]
        )
        chunks = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        vectorstore.save_local(self.config["index_path"])
        return vectorstore

    def _get_loader(self, source):
        """Factory method for loaders"""
        if source["type"] == "pdf":
            return PyPDFLoader(source["path"])
        elif source["type"] == "web":
            return WebBaseLoader(source["path"])
        elif source["type"] == "folder":
            return DirectoryLoader(
                source["path"],
                glob="**/*.md",
                loader_cls=UnstructuredMarkdownLoader
            )
        elif source["type"] == "csv":
            return CSVLoader(source["path"])
        else:
            raise ValueError(f"Unsupported source type: {source['type']}")
    
    def retrieve(self, query, k=3):
        """Retrieve relevant documents"""
        return self.vectorstore.similarity_search(query, k=k)
    
    def generate(self, query, retrieved_docs):
        """Generate response using LLM"""
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        messages = [
            {"role": "system", "content": "Answer the question based only on the following context:"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = self.llm.pipeline(
            prompt,
            **self.config["generation_config"]
        )
        
        return outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
    
    def query(self, question, k=3):
        """End-to-end RAG query"""
        retrieved_docs = self.retrieve(question, k)
        return self.generate(question, retrieved_docs)
    
    def cleanup(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared")

def main():
    config = {
        "data_sources": [
            {"type": "pdf", "path": "data/deccanqueen.pdf"},
            {"type": "pdf", "path": "data/pune.pdf"}
            #{"type": "web", "path": "https://company.com/docs"},
            #{"type": "folder", "path": "data/knowledge_base/"}
        ],
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "HuggingFaceH4/zephyr-7b-alpha",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "index_path": "faiss_index",
        "load_existing_index": False,
        "hf_home": "D:/huggingface_models/huggingface",
        "generation_config": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            "do_sample": True
        }
    }

    # Initialize system once
    rag = RAGSystem(config)
    
    # Interactive query loop
    print("\nRAG System ready! Type 'exit' to quit.")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() in ['exit', 'quit']:
            break
        
        response = rag.query(question)
        print(f"\nAnswer: {response}")
    
    # Cleanup
    rag.cleanup()
    print("Session ended.")

if __name__ == "__main__":
    main()