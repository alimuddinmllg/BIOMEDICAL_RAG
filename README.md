# Medical Document Q&A System with RAG 🏥📄

![System Architecture](https://via.placeholder.com/Screenshot 2025-03-07 at 09.35.19.png)

A Retrieval-Augmented Generation (RAG) system for querying medical literature using PubMedBERT embeddings and BioMistral-7B LLM.

## Features ✨

- **Automated PDF Processing**: Bulk import from directory
- **Domain-Specific Embeddings**: PubMedBERT for medical context
- **Clinical Response Generation**: BioMistral-7B fine-tuned LLM
- **Semantic Search**: Qdrant vector database integration
- **Interactive Interface**: Streamlit web UI

## Tech Stack 🛠️

| Component               | Technology                          |
|-------------------------|-------------------------------------|
| Language Model          | BioMistral-7B-Q4_K_M                |
| Text Embeddings         | PubMedBERT-base-embeddings          |
| Vector Database         | Qdrant                              |
| Framework               | LangChain                           |
| UI Framework            | Streamlit                           |
| PDF Processing          | PyMuPDF                             |

## Installation 💻

### Prerequisites
- Python 3.8+
- Docker
- 8GB+ RAM (16GB Recommended)
- x86-64 CPU with AVX2 support

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/medical-rag-system.git
cd medical-rag-system

2. Install Dependencies
pip install -r requirements.txt

3. Start Qdrant Service
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

4. run main_file.py
streamlit run main_file.py

