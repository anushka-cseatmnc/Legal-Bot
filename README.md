# 🧑‍⚖️ LegalBot – AI-Powered Legal Rights Chatbot
Gen AI + RAG + Web Scraping + LangChain

# 🔍 Overview
LegalBot is an AI-powered conversational web app that helps individuals understand their legal rights in real-time. Using LLMs, Retrieval-Augmented Generation (RAG), and web-scraped legal data, the chatbot provides personalized, accurate, and contextual legal information through a friendly Streamlit interface.

# 🚨 Problem Statement
Accessing legal information is difficult for the average citizen. Law is buried in dense documents, official portals are hard to navigate, and human lawyers are expensive. Many individuals aren't aware of their basic rights — such as consumer protection, FIR filing, or digital privacy.

# 💡 Solution
LegalBot simplifies legal knowledge through conversational AI. By scraping verified government portals and feeding the content into a vector database, the system enables an LLM to provide accurate, up-to-date, and contextually relevant answers.

The chatbot can address queries like-
What are my rights when I get arrested?
How can I file a complaint about online fraud?
Is it legal for employers to withhold salary?

# 🚀 Current Progress

✅ Features Implemented

🔗 Web Scraper -  Extracts legal text from official Indian government and law commission websites.

🧠 RAG Pipeline - Integrates FAISS vector store + LangChain + LLM to generate factually relevant answers.

📚 Legal Domains - Covers rights related to FIRs, consumer complaints, cyber law, and constitutional protections.

💬 Streamlit Chat UI- A responsive frontend that allows users to chat naturally with the AI.

⚙️ FastAPI Backend- Handles prompt routing, LLM interaction, and response delivery via REST endpoints.

# 🧞‍♂️ How It Works

User Enters a Legal Question
        ↓
FastAPI Receives Query → Sentiment & Prompt Cleaning (optional)
        ↓
Vector DB (FAISS) Queries Relevant Context From Scraped Data
        ↓
LangChain Combines Prompt + Context → Sent to LLM (OpenAI/Mistral)
        ↓
LLM Generates an Informed, Contextual Answer
        ↓
Streamlit Displays the Reply in a Conversational UI

# 🔥 Upcoming Features

 User Authentication – Login/Signup system to store chat history.

 Conversation Memory – Enable follow-up questions via session memory.

 Voice-to-Text Input – Let users speak their queries.

 Law Domain Expansion – Add property rights, marriage laws, tenant laws, etc.

 PDF Upload – Summarize or extract rights from personal legal documents.

 Multilingual Support – Serve Hindi, Tamil, and other Indian languages.

 Case Law Generator – Suggest recent legal precedents related to user queries.


# 🛠️ Tech Stack

 Frontend	Streamlit (Chat UI)
 Backend	FastAPI (API Server)
LLM	OpenAI GPT-4 / Groq LLaMA / Mistral
Framework	LangChain
Vector DB	FAISS + OpenAIEmbeddings
Scraping	BeautifulSoup / Requests (Gov legal sites)
Deployment	Streamlit Cloud / Render / HuggingFace Spaces

 Example Queries-
Query	Sample Bot Response
Can I file an FIR online?	Yes. In most Indian states, FIRs can be filed online via police portals.
What happens if I’m detained without a warrant?	Under Article 22, you're entitled to legal counsel and must be presented to a magistrate within 24 hours.
Are unsolicited promotional calls legal? As per TRAI, you can register under DND to prevent these. Violations can be reported.

# 📂 Project Structure
graphql
Copy
Edit
LegalBot/
│
├── backend/
│   ├── main.py                # FastAPI backend
│   ├── rag_pipeline.py        # LangChain + FAISS logic
│   ├── scraper.py             # Web scraper for legal docs
│   ├── utils.py               # Helper functions
│   └── requirements.txt       # Backend dependencies
│
├── frontend/
│   ├── app.py                 # Streamlit chatbot interface
│   ├── components.py          # Custom UI components
│   └── requirements.txt       # Frontend dependencies
│
├── data/
│   └── legal_docs/            # Raw + processed legal text
│
├── models/
│   └── vector_store.pkl       # FAISS vector store (or Chroma)
│
├── .env                       # API keys and config
├── README.md                  # You’re reading this 📘
└── run.sh                     # Launches backend + frontend

# 📌 Next Steps
🔹 Train and integrate a summarization LLM for long queries

🔹 Deploy on HuggingFace Spaces / Streamlit Cloud

🔹 Add basic legal document upload and question answering from docs

🔹 Add privacy policy and disclaimer module

🔹 Improve prompt templating and hallucination control

# 🚀 Why It Matters

Legal Awareness at Scale – Helps citizens understand their rights instantly.

Cost-Effective – Replaces need for preliminary legal consultations.

Educational – Promotes civic knowledge and empowerment.

Scalable – Adaptable to different countries, jurisdictions, and use cases.

# 🌍 Real-World Impact
✅ Equips citizens with legal knowledge
✅ Reduces dependency on search-based research
✅ Makes legal info accessible to rural/underserved regions
✅ Potential use by NGOs, legal aid groups, educational institutions

📜 License
This project is licensed under the MIT License. Refer to the LICENSE file for more info.

👨‍💻 Author
Anushka Chaudhary
Integrated M.Tech (CSE, AI) '28
Passionate about GenAI, System Design & Legal Tech

## "Access to justice begins with access to information."












Tools



ChatGPT can make mistakes. Check important info. See Cookie Preferences.
