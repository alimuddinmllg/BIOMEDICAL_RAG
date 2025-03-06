from langchain_community.llms.llamacpp import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain.llms.base import LLM
from typing import Optional
from pydantic.fields import Field
from fastapi import Body, HTTPException
import os
import json
from ctransformers import AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory = 'templates')

app.mount("/static", StaticFiles(directory="static"), name="static")

# Custom wrapper for ctransformers LLM
class CustomCTransformersLLM(LLM):
    model: object = Field(..., exclude=True)  # Declare `model` as a Pydantic field


    @property
    def _llm_type(self):
        return "custom_ctransformers"

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        response = self.model(prompt)
        if stop:
            for stop_token in stop:
                response = response.split(stop_token)[0]
        return response
    

model_name = "MaziyarPanahi/BioMistral-7B-GGUF"  # Update this with the actual repository
model_file = "BioMistral-7B.Q4_K_M.gguf"  # The filename you need

print("Downloading model if not available...")
model_path = hf_hub_download(repo_id=model_name, filename=model_file)

# Load the model
llm = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama", gpu_layers=0)

# Wrap in Custom LLM
custom_llm = CustomCTransformersLLM(model=llm)

print("LLM Initialized successfully!")

# local_llm = "BioMistral-7B.Q4_K_M.gguf"
# if os.path.exists(local_llm):
#     model_path = os.path.abspath(local_llm)
#     print(f"Model exists at: {model_path}")
#     llm = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         model_type="llama",
#         gpu_layers=0  # Set to 0 for CPU
#     )
#     custom_llm = CustomCTransformersLLM(model=llm)
# else:
#     raise FileNotFoundError(f"Model file not found at {local_llm}")


# print ("LLM Initilized.....")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

embeddings = HuggingFaceEmbeddings(model_name="neuml/pubmedbert-base-embeddings")

url = "http://localhost:6333"

client = QdrantClient(
    url = url, prefer_grpc= False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

prompt = PromptTemplate(template = prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs = {"k":2})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})




@app.post("/get_response")
async def get_response(payload: dict = Body(...)):
# async def get_response(query: str = Body(..., embed=True)):
    try:
        # Validation moved inside the endpoint
        query = payload.get("query")
        if not query or len(query.strip()) < 3:
            raise HTTPException(status_code=400, detail="Query must be at least 3 characters")

        print(f"Received query: {query}")
        chain_type_kwargs = {"prompt": prompt}
        qa = RetrievalQA.from_chain_type(
            llm=custom_llm,  # Use the wrapped LLM
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True,
            chain_type_kwargs=chain_type_kwargs
        )
        print("RetrievalQA initialized")
        response = qa(query)
        print(f"Response generated: {response}")

        # Extract result components
        answer = response.get("result", "No answer found")
        source_documents = response.get("source_documents", [])

        # Prepare source details
        if source_documents:
            source_document = source_documents[0].page_content
            doc_metadata = source_documents[0].metadata.get("source", "Unknown Source")
        else:
            source_document = "No relevant documents found"
            doc_metadata = "Unknown Source"

        # Create JSON response
        response_data = {
            "answer": answer,
            "source_document": source_document,
            "doc": doc_metadata,
        }
        return Response(content=json.dumps(response_data), media_type="application/json")

    except HTTPException as he:
        raise he  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"Error processing query: {e}")  # Log first
        raise HTTPException(status_code=500, detail=str(e))