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

### Data Flow

![image](https://github.com/user-attachments/assets/abd3ee0c-25e4-4e09-8de3-5929e5c2d433)

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

  ### RAG Pipeline 
- ![image](https://github.com/user-attachments/assets/d7b949d2-972f-48f5-9fed-e4c72f1b15be)


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


 ## Open AI API LOGIC is integrated in RAG --
 
>This project uses a Retrieval-Augmented Generation (RAG) pipeline designed for resilience and privacy. While OpenAI’s gpt-3.5-turbo is integrated for inference, the system defaults to a local LLM if-
>OpenAI quota is exhausted
> The API key is missing or invalid
>The OpenAI service is unreachable at runtime

## Chain of Logic

> At Startup of application 
> Both OpenAI and local LLM-based RAG chains are initialized.
>Local LLM is warmed up with a dummy invoke("hello") call to reduce latency.
> During Runtime (Query Inference):
>The system first attempts to respond using the OpenAI RAG chain.
>If that fails (e.g., RateLimitError, timeout), it automatically falls back to the local chain—no crash, no user-facing error.
>How Fallback Works:
> A helper function safe_invoke() wraps the .invoke() call.
> Any exception from OpenAI inference is caught.
>  The local RAG pipeline is used to generate a final response

---
### Why This App Can’t Be Deployed on Free Cloud Tiers

Although this project runs completely offline and doesn’t depend on external APIs, it requires a GPU, enough memory, and persistent storage to work properly. Free cloud platforms like Render, Hugging Face Spaces, Heroku, or AWS Free Tier don’t provide enough resources to host this app. Because 

- **GPU is Required**: Even with compression, the Mistral 7B model still needs a GPU for fast, stable responses. Free services don’t offer this.
- **Not Enough RAM**: Running the model and ChromaDB together needs –16 GB RAM, far more than the ~1 GB free tiers allow.
- **Storage Limitations**: The model weights (~4GB) and vector database (~5–10 GB) won’t fit within the tiny storage limits of free hosting plans.
- **Slow and Unreliable**: Free tiers often pause idle apps or limit connections, causing delays and breaking the experience for users needing emotional support.

### 💡 Why I Didn’t Use a Smaller Language Model to make it feasible for deployment
- Smaller models (like 1B–3B) were tested but didn’t perform well.
- They couldn’t handle emotional context accurately or give empathetic responses.
- They also struggled in combining retrieved information from documents, which is key in a RAG setup.
- 
> That’s why I chose the 7B model—it gives better, safer, and more human-like support without relying on cloud APIs.

### Live Demo & System Walkthrough

To demonstrate the app’s capabilities without requiring live hosting, a recorded demo video is provided. It includes:
- Full walkthrough of the codebase
- Explanation of the system architecture
- Discussion of scalability considerations
- Identification of bottlenecks and trade-offs
> 📌 Watch the Demo Video ( will be avilable soon )
> screenshot for refrence
> ![WhatsApp Image 2025-07-04 at 21 17 22_0424106c](https://github.com/user-attachments/assets/2843700a-b317-4b1c-a18d-3b1127274ce2)

## 🧠 Future Enhancements

- Voice Input (Speech-to-Text)
- Multilingual LegalBot (Hindi, Tamil, Marathi)
- UI Chat History + Login
- Legal News & Updates Feed
- PDF Upload for personal documents

 ### Legal-Bot Local Installation 

 1-System Requirements
Python 3.8+ , 8GB+ RAM (for Mistral-7B model)


```1. Clone & Setup
bashgit clone https://github.com/anushka-cseatmnc/Legal-Bot.git
cd Legal-Bot```

```2. Install Dependencies
bash# Install backend dependencies
cd backend
pip install -r requirements.txt```

# Install frontend dependencies
```cd ../frontend
pip install -r requirements.txt```

3. Download Mistral-7B Model
```bashmkdir models
Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
File: mistral-7b-instruct-v0.1.Q4_K_M.gguf (~4GB)
 Save to: models/ folder```

4. Initialize Vector Database
```bashcd backend
python ingest.py  # Processes all 24 legal PDFs (takes 10-15 mins)```

5. Run the Application
Option 1: Using run.sh (Recommended)
```bashchmod +x run.sh
./run.sh```
Option 2: Manual startup
bash# Terminal 1 - Backend
``cd backend
uvicorn main:app --host 0.0.0.0 --port 8000``

# Terminal 2 - Frontend  
```cd frontend
streamlit run app.py --server.port 8501
```


### Access Points
Chat Interface: http://localhost:8501
API Docs: http://localhost:8000/docs

⚠Quick Troubleshooting

Memory Error: Ensure 8GB+ RAM available
Model Not Found: Check model file path in rag_pipeline.py
Port Busy: Kill existing processes or use different ports
PDF Issues: Verify all 24 PDFs are in data/legal_docs/

 Test Your Setup
Ask: "What are fundamental rights in India?"


## 👩‍💻 Built By

**Anushka Chaudhary**  
Integrated M.Tech (CSE, AI) | VIT Bhopal  
 GenAI • LegalTech • LLM • LangChain • ML

---

> _"Empowering people with legal awareness is the first step toward justice."_  
> _Be aware. Be informed. Be legally secure._

---









