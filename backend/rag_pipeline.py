# rag_pipeline.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers  

def get_rag_chain():
    db = Chroma(persist_directory="./chroma_db", embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    retriever = db.as_retriever()

    custom_prompt_template = """
    You are an expert legal assistant. Use only the following retrieved documents to answer the legal query.
    Be precise, fact-based, and never guess. If you can’t find a direct answer, say:
    “Sorry, I couldn't find relevant information in the provided legal documents.”

    Context from Documents:
    -----------------------
    {context}

    Question: {question}
    -----------------------
    Answer (cite legal points if needed):
    """

    PROMPT = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

    llm = CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        model_type="mistral",
        max_new_tokens=512,
        temperature=0.1
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return chain
