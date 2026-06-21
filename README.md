# AI Medical Receptionist Agent — Setup Guide

## Project Structure
```
medical-agent/
├── backend/
│   ├── main.py              ← FastAPI app
│   ├── agents/
│   │   └── medical_agent.py ← AI Agent (LangChain + HuggingFace)
│   ├── rag/
│   │   └── rag_engine.py    ← FAISS + RAG
│   ├── database/
│   │   ├── database.py      ← DB config
│   │   └── models.py        ← SQLAlchemy models
│   └── data/
│       ├── csv/             ← All CSV files
│       └── pdf/             ← PDF + TXT knowledge base
├── frontend/
│   └── src/
│       ├── App.js           ← React UI
│       └── styles/App.css   ← Styles
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## Local Setup

### Step 1 — Clone & Install
```bash
git clone <your-repo>
cd medical-agent
pip install -r requirements.txt
```

### Step 2 — Setup .env
```bash
cp .env.example .env
# Edit .env with your DB credentials
```

### Step 3 — Run Backend
```bash
uvicorn backend.main:app --reload --port 8000
```

### Step 4 — Run Frontend
```bash
cd frontend
npx create-react-app .
cp src/App.js src/styles/App.css src/
npm start
```

---

## Render Deployment

1. Push to GitHub
2. Go to render.com → New Web Service
3. Connect GitHub repo
4. Set Build Command: `pip install -r requirements.txt`
5. Set Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables from .env.example
7. Attach PostgreSQL database
8. Deploy!

---

## API Endpoints

| Endpoint       | Method | Description              |
|----------------|--------|--------------------------|
| /chat          | POST   | Text chat with AI agent  |
| /voice-chat    | POST   | Voice chat (audio upload)|
| /doctors       | GET    | Get all doctors          |
| /beds          | GET    | Get bed availability     |
| /departments   | GET    | Get departments          |
| /pharmacy      | GET    | Get pharmacy inventory   |
| /patients      | GET    | Get patient list         |
| /appointments  | GET    | Get appointments         |
| /billing       | GET    | Get billing records      |
| /staff         | GET    | Get staff list           |
| /health        | GET    | Health check             |

---

## Sample Questions

- "Which doctors are available today?"
- "Show cardiology doctors"
- "How many ICU beds are available?"
- "What are the hospital rules?"
- "What are visiting hours?"
- "Show pharmacy inventory"
- "Which department handles heart disease?"
- "Show emergency contacts"
