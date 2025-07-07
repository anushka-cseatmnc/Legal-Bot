# rag_pipeline.py - Improved version with OpenAI integration and better prompts
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain.schema import BaseRetriever
from typing import Optional
import logging
import requests 

def openai_quota_available() -> bool:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    try:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        resp = requests.get("https://api.openai.com/dashboard/billing/credit_grants", headers=headers)
        data = resp.json()
        return data.get("total_available", 0) > 0
    except Exception as e:
        logger.warning(f"OpenAI quota check failed: {e}")
        return False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGConfig:
    def __init__(self):
        self.use_openai = bool(os.getenv("OPENAI_API_KEY"))
        self.temperature = 0.1
        self.max_tokens = 256
        self.chunk_size = 300
        self.chunk_overlap = 100
        self.retrieval_k = 2  # Reduced from 6-8 for better performance

def create_enhanced_prompt():
    """Create an improved prompt template for legal queries"""
    
    prompt_template = """You are an expert Indian legal assistant with deep knowledge of Indian law. 
       Based ONLY on the provided legal documents, answer the question comprehensively and accurately.
      If the user question is general (not legally technical), provide a concise summary in 4–6 lines.
      Mention legal acts only if relevant, and avoid repeating generic definitions. Be clear, accurate, and user-friendly.

INSTRUCTIONS:
- Give a complete, well-structured answer
- If the question has multiple parts, address each part clearly
- Use numbered points or bullet points when listing rights/procedures
- Cite specific legal provisions when mentioned in the documents
- Use professional legal language but ensure clarity

CONTEXT FROM LEGAL DOCUMENTS:
{context}

QUESTION: {question}

COMPREHENSIVE LEGAL ANSWER:"""
    
    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

def create_openai_chain(config: RAGConfig) -> Optional[RetrievalQA]:
    """Create RAG chain using OpenAI API - returns None if fails (no errors)"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.info(" No OpenAI API key found, will use local LLM")
            return None
            
        logger.info(" OpenAI API key found, initializing OpenAI chain...")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load vector database
        db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        
        # Configure retriever with better parameters
        retriever = db.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={
                "k": config.retrieval_k,
                "fetch_k": config.retrieval_k * 2,
                "lambda_mult": 0.7  # Balance between similarity and diversity
            }
        )
        
        # Initialize OpenAI LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Fast and cost-effective
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=api_key
        )
        
        # Create enhanced prompt
        prompt = create_enhanced_prompt()
        
        # Create the chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=False  # Reduced verbosity
        )
        
        logger.info(" OpenAI RAG chain created successfully")
        return chain
        
    except Exception as e:
        logger.info(f" OpenAI setup failed ({str(e)}), will use local LLM")
        return None

def create_local_chain(config: RAGConfig) -> Optional[RetrievalQA]:
    """Create RAG chain using local LLM as fallback"""
    try:
        logger.info("Creating local LLM-based RAG chain...")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load vector database
        db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        
        # Configure retriever
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.retrieval_k,
                "fetch_k": config.retrieval_k * 2
            }
        )
        
        # Initialize local LLM with optimized settings
        llm = CTransformers(
            model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            model_type="mistral",
            max_new_tokens=1024,
            temperature=0.1,
            # Optimization parameters
            threads=6,  # Use multiple threads
            gpu_layers=0,  # Set to >0 if you have GPU
            context_length=2048
        )
        
        # Create enhanced prompt for local LLM
        prompt = create_enhanced_prompt()
        
        # Create the chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        logger.info(" Local LLM RAG chain created successfully")
        return chain
        
    except Exception as e:
        logger.error(f" Failed to create local chain: {e}")
        return None

def get_rag_chain(use_openai: bool = None) -> RetrievalQA:
    """
    Get the RAG chain with automatic fallback logic - NO ERRORS!
    
    Logic: OpenAI API if key found → Local LLM (always works)
    
    Args:
        use_openai: Force OpenAI usage (True) or local LLM (False). 
                   If None, auto-detect based on API key availability.
    """
    config = RAGConfig()
     # Automatically detect availability
    if use_openai is None:
        config.use_openai = openai_quota_available()

    
    # Override config if explicitly specified
    if use_openai is not None:
        config.use_openai = use_openai
    
    # Try OpenAI first if API key is available
    if config.use_openai:
        logger.info(" OpenAI API key found, using OpenAI...")
        try:
            chain = create_openai_chain(config)
            if chain:
                logger.info(" OpenAI chain created successfully")
                return chain
        except Exception as e:
            logger.warning(f" OpenAI failed ({e}), falling back to local LLM...")
    
    # Always fallback to local LLM (this should never fail)
    logger.info(" Using local LLM...")
    try:
        chain = create_local_chain(config)
        if chain:
            logger.info(" Local LLM chain created successfully")
            return chain
    except Exception as e:
        logger.error(f" Local LLM also failed: {e}")
        # Last resort - create a minimal working chain
        return create_minimal_chain(config)
    
    # This should never be reached
    raise RuntimeError(" All chain creation methods failed")



def safe_invoke(primary_chain: RetrievalQA, question: str, fallback_chain: Optional[RetrievalQA] = None):
    try:
        return primary_chain.invoke(question)
    except Exception as e:
        logger.warning(f"Primary chain failed during invoke: {e}")
        if fallback_chain:
            logger.info("Trying fallback chain...")
            return fallback_chain.invoke(question)
        raise e 

def test_chain_performance():
    """Test the performance of both chains"""
    import time
    
    test_question = "What are the fundamental rights of Indian citizens?"
    
    print(" Testing RAG Chain Performance...")
    
    # Test OpenAI chain
    try:
        start_time = time.time()
        openai_chain = get_rag_chain(use_openai=True)
        if openai_chain:
            result = openai_chain.invoke(test_question)
            openai_time = time.time() - start_time
            print(f"OpenAI Response Time: {openai_time:.2f} seconds")
        else:
            print(" OpenAI chain not available")
    except Exception as e:
        print(f"OpenAI test failed: {e}")
    
    # Test local chain
    try:
        start_time = time.time()
        local_chain = get_rag_chain(use_openai=False)
        result = local_chain.invoke(test_question)
        local_time = time.time() - start_time
        print(f" Local LLM Response Time: {local_time:.2f} seconds")
    except Exception as e:
        print(f" Local test failed: {e}")

if __name__ == "__main__":
    # Test the chains
    test_chain_performance()
    
    # Create default chain
    chain = get_rag_chain()
    print(" RAG chain ready for use!")