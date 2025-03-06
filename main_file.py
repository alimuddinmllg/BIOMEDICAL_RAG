import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from ctransformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from qdrant_client import QdrantClient
from langchain_core.runnables import RunnableSerializable
from typing import Any, Optional, List
from langchain_core.prompt_values import StringPromptValue

# Set up Streamlit UI
st.set_page_config(page_title="AI-Powered PDF Q&A", layout="wide")
st.title("📄 AI-Powered PDF Q&A with Qdrant")

# Model & Embeddings
embeddings = HuggingFaceEmbeddings(model_name="neuml/pubmedbert-base-embeddings")
model_name = "MaziyarPanahi/BioMistral-7B-GGUF"
model_file = "BioMistral-7B.Q4_K_M.gguf"

# Qdrant setup
url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)

# Load and process PDFs from data directory
data_dir = "data"
if os.path.exists(data_dir) and os.path.isdir(data_dir):
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", show_progress=True)
    documents = loader.load()
    
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
        texts = text_splitter.split_documents(documents)
        
        # Store in Qdrant
        qdrant = Qdrant.from_documents(
            texts,
            embeddings,
            url=url,
            prefer_grpc=False,
            collection_name="vector_db"
        )
        st.sidebar.success(f"Successfully indexed {len(documents)} PDF documents!")
    else:
        st.sidebar.warning("No PDF files found in the 'data' directory")
else:
    st.sidebar.error("'data' directory not found. Please create it and add PDF files.")

# Load LLM
model_path = hf_hub_download(repo_id=model_name, filename=model_file)

# Custom LLM Wrapper
class CustomCTransformersLLM(RunnableSerializable):
    def __init__(self, model_path):
        super().__init__()
        object.__setattr__(self, "model", 
            AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="llama",
                gpu_layers=0
            )
        )

    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs) -> str:
        if isinstance(input, StringPromptValue):
            text_input = input.to_string()
        else:
            text_input = str(input)

        stop = kwargs.get("stop", None)
        response = self.model(text_input)
        
        if stop:
            for stop_token in stop:
                response = response.split(stop_token)[0]
        return response

llm = CustomCTransformersLLM(model_path)

# Initialize Qdrant retriever
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")
retriever = db.as_retriever(search_kwargs={"k": 2})

# Prompt template
prompt_template = """Use the following pieces of information to answer the user's question.
Context: {context}
Question: {question}
Only return the helpful answer.
Helpful answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Query interface
st.subheader("Ask a Question from PDF Documents")
query = st.text_input("Enter your query:")
if query:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    response = qa(query)
    
    st.subheader("Answer")
    st.write(response.get("result", "No answer found"))
    
    st.subheader("Source Document")
    source_doc = response.get("source_documents", [])
    if source_doc:
        st.write(source_doc[0].page_content)
    else:
        st.write("No relevant documents found.")