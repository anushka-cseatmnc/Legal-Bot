import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGSettings:
    # LLM Settings
    use_openai: bool = False
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    
    # Local LLM Settings
    local_model_path: str = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    local_model_file: str = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    
    # Common Settings
    temperature: float = 0.1
    max_tokens: int = 1024
    
    # Vector DB Settings
    chroma_db_path: str = "./chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Retrieval Settings
    retrieval_k: int = 8
    retrieval_fetch_k: int = 20
    
    @classmethod
    def from_env(cls):
        """Load settings from environment variables"""
        return cls(
            use_openai=os.getenv("USE_OPENAI", "false").lower() == "true",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("MAX_TOKENS", "1024")),
        )