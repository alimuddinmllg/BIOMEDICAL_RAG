# Medical Document Q&A System with RAG 🏥📄

![Image Description](https://raw.githubusercontent.com/alimuddinmllg/BIOMEDICAL_RAG/main/Image1.png)


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
git clone https://github.com/alimuddinmllg/medical-rag-system.git
cd medical-rag-system
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Start Qdrant Service
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```
5. run main_file.py
```bash
streamlit run main_file.py
```

