from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_documents(data_path='data/'):
    docs = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, filename))
            docs.extend(loader.load())
    return docs

def create_vector_store():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(split_docs, OpenAIEmbeddings(), persist_directory="./chroma_db")
    vectorstore.persist()
    return vectorstore
