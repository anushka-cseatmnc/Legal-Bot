# ⚖️LegalBot – AI-Powered Legal Rights Chatbot

> “Access to justice begins with access to information.”

---

## 🔍 Overview

**LegalBot** is a RAG-based legal assistant that answers Indian law-related queries using a curated set of official legal PDFs. It uses local embeddings and local LLMs (fully offline), making it cost-effective and private.

---

## 🚨 Problem Statement

Most people struggle with legal awareness due to:
- Legalese-heavy documents
- Costly consultations
- Low awareness of fundamental rights

---

## 💡 Solution

LegalBot solves this with:
-  Local **RAG Pipeline** using LangChain + FAISS + Mistral 7B
-  24 Official Indian Legal PDFs as the core knowledge base
- Streamlit Chat UI with FastAPI backend
- Offline, privacy-respecting architecture (no API keys!)

---

## ⚙️ Tech Stack

| Layer       | Tools Used |
|-------------|------------|
| Embedding   | `HuggingFaceEmbeddings` (MiniLM) |
| Vector DB   | `Chroma` (FAISS backend) |
| Document    | `LangChain PyPDFLoader`, `TextSplitter` |
| LLM         | `Mistral-7B-Instruct` (GGUF) via `ctransformers` |
| Backend     | `FastAPI` |
| Frontend    | `Streamlit` |
| Prompting   | `LangChain PromptTemplate` |

---

## 🧠 Technical Architecture

### 1. `ingest.py` – Document Preprocessing & Vector Storage
- Loads 24 PDFs
- Splits into 1000-character chunks with 200 overlap
- Embeds using MiniLM → stores in persistent FAISS (Chroma)

### 2. `rag_pipeline.py` – RAG Chain Construction
- Loads FAISS vector DB
- Loads quantized Mistral-7B using `CTransformers`
- Custom Prompt ensures:
  - Only relevant legal info
  - Fallback if answer not found

### 3. `main.py` – FastAPI Backend
- POST `/chat` → gets legal answer
- POST `/query` → gets answer + cited pages
- GET `/health` → for uptime check

---

## 📁 Directory Structure
LegalBot/
├── backend/
│ ├── ingest.py # Vector store creation
  ├── db.py 
│ ├── rag_pipeline.py # RAG chain logic
│ └── main.py # FastAPI server
├── frontend/
│ └── app.py # Streamlit chat UI
├── data/legal_docs/ # 24 Legal PDFs
└── README.md # You're here!

---

## 🧾 List of Legal PDFs Used (All 24)

1. constitution of india.pdf  
2. Indian Penal Code (IPC).pdf  
3. RTI Act, 2005.pdf  
4. Digital Rights & Privacy.pdf  
5. Filing an FIR.pdf  
6. Dowry_prohibition_act.pdf  
7. Divorce act.pdf  
8. Special Marriage Act.pdf  
9. The Hindu Marriage Act.pdf  
10. POSH Act (Sexual Harassment).pdf  
11. Human Rights.pdf  
12. Women’s Rights in India.pdf  
13. Juvenile Justice Act.pdf  
14. SC-ST Atrocities Act.pdf  
15. Muslim Personal Law.pdf  
16. Indian Contract Act.pdf  
17. Indian Majority Act.pdf  
18. Consumer protection.pdf  
19. Fundamental rights.pdf  
20. Right to Education Act.pdf  
21. 11_IT_and_Cyber_Laws.pdf  
22. Labor laws in India.pdf  
23. NRI legal rights.pdf  
24. How to file complaint in cyber cell.pdf  

---

## 💬 Sample Questions It Can Handle

- “What are my rights if police deny an FIR?”
- “How to file a cyber crime complaint?”
- “Is dowry punishable in India?”
- “What are labor laws for working women?”
- “What is the minimum age for marriage under Hindu law?”

---

## 🛡️ Privacy First

✅ No OpenAI API keys  
✅ No user tracking  
✅ 100% offline capability  
✅ Local LLM + Vector DB  

---

## 🧠 Future Enhancements

- Voice Input (Speech-to-Text)
- Multilingual LegalBot (Hindi, Tamil, Marathi)
- UI Chat History + Login
- Legal News & Updates Feed
- PDF Upload for personal documents

---

## 👩‍💻 Built By

**Anushka Chaudhary**  
Integrated M.Tech (CSE, AI) | VIT Bhopal  
 GenAI • LegalTech • LLM • LangChain • ML

---

> _"Empowering people with legal awareness is the first step toward justice."_  
> _Be aware. Be informed. Be legally secure._

---









