# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from backend.rag_pipeline import get_rag_chain
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
qa_chain = get_rag_chain()

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
    result = qa_chain.invoke(req.question)
    return {"response": result["result"]}
@app.post("/query/")
async def query(req: QueryRequest):
    result = qa_chain.invoke(req.question)
    return {"answer": result["result"], "source_documents": [doc.page_content for doc in result["source_documents"]]}
@app.get("/")
async def root():
    return {"message": "Welcome to the Legal Assistant API. Use /chat or /query endpoints."}
@app.get("/health")
async def health():
    return {"status": "ok"}
@app.get("/docs")
async def docs():
    return {"message": "API documentation is available at /docs or /redoc."}