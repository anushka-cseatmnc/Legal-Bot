# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from backend.rag_pipeline import get_rag_chain, safe_invoke 
from fastapi.middleware.cors import CORSMiddleware
import time


app = FastAPI()
openai_chain = get_rag_chain(use_openai=True)
local_chain = get_rag_chain(use_openai=False)

# Warm up the local model once at app startup
try:
    _ = local_chain.invoke("hello")  # dummy run to reduce cold-start latency
except Exception as e:
    print(f"Local model warm-up failed: {e}")


class QueryRequest(BaseModel):
    question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or set to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(req: QueryRequest):
    result = safe_invoke(openai_chain, req.question, local_chain)
 
    return {"response": result["result"]}

@app.post("/query/")
async def query(req: QueryRequest):
    start = time.time()
    result = safe_invoke(openai_chain, req.question, local_chain)
    end = time.time()
    
    print(f" Response time: {end - start:.2f} sec")

    for i, doc in enumerate(result["source_documents"]):
        print(f"\n Retrieved Document {i+1}:\n{doc.page_content[:300]}...\n")

    return {
        "answer": result["result"],
        "source_documents": [doc.page_content for doc in result["source_documents"]],
        "response_time_sec": round(end - start, 2)
    }
@app.get("/")
async def root():
    return {"message": "Welcome to the Legal Assistant API. Use /chat or /query endpoints."}
@app.get("/health")
async def health():
    return {"status": "ok"}
@app.get("/docs")
async def docs():
    return {"message": "API documentation is available at /docs or /redoc."}

