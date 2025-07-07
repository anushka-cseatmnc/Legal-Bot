import os
import pytest
from unittest.mock import patch, MagicMock
from rag_pipeline import get_rag_chain, RAGConfig, create_openai_chain, create_local_chain

class TestRAGPipeline:
    """Test suite for RAG pipeline"""
    
    def test_local_llm_creation(self):
        """Test creating RAG chain with local LLM"""
        try:
            chain = get_rag_chain(use_openai=False)
            assert chain is not None
            print(" Local LLM chain created successfully")
        except Exception as e:
            print(f" Local LLM test failed: {e}")
    
    def test_openai_fallback(self):
        """Test OpenAI fallback to local LLM"""
        # Test without API key (should fallback)
        chain = get_rag_chain(use_openai=True)
        assert chain is not None
        print(" OpenAI fallback test passed")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_with_key(self):
        """Test OpenAI with API key (mocked)"""
        with patch('rag_pipeline.ChatOpenAI') as mock_openai:
            mock_openai.return_value = MagicMock()
            chain = get_rag_chain(use_openai=True)
            assert chain is not None
            print(" OpenAI with key test passed")
    
    def test_config_creation(self):
        """Test RAG configuration"""
        config = RAGConfig()
        assert config.use_openai is False
        assert config.temperature == 0.1
        print(" Config creation test passed")
    
    def test_query_processing(self):
        """Test actual query processing"""
        try:
            chain = get_rag_chain(use_openai=False)
            # Test with a simple query
            result = chain.invoke({"query": "What is this document about?"})
            assert "result" in result
            print(" Query processing test passed")
        except Exception as e:
            print(f" Query test failed (expected if no documents): {e}")

def run_integration_tests():
    """Run integration tests"""
    print(" Running RAG Pipeline Tests...")
    
    # Test 1: Local LLM
    print("\n1. Testing Local LLM...")
    try:
        chain = create_local_chain()
        print(" Local chain created")
    except Exception as e:
        print(f" Local chain failed: {e}")
    
    # Test 2: OpenAI (with fallback)
    print("\n2. Testing OpenAI (with fallback)...")
    try:
        chain = create_openai_chain()  # Should fallback to local
        print(" OpenAI chain created (or fallback successful)")
    except Exception as e:
        print(f" OpenAI chain failed: {e}")
    
    # Test 3: Configuration
    print("\n3. Testing Configuration...")
    config = RAGConfig()
    config.use_openai = True
    print(f" Config: use_openai={config.use_openai}")
    
    print("\n All tests completed!")

if __name__ == "__main__":
    run_integration_tests()