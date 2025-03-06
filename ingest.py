import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredAPIFileIOLoader, DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="neuml/pubmedbert-base-embeddings")

loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=PyMuPDFLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap=70)

texts = text_splitter.split_documents(documents)

url = "http://localhost:6333"

qdrant = Qdrant.from_documents(
    texts, 
    embeddings,
    url=url,
    prefer_grpc = False,
    collection_name = "vector_db"
)

print ("Vector DB is")