import os
import logging
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
# import whisper

from backend.agents.medical_agent import MedicalReceptionistAgent

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Init ──────────────────────────────────────────────────
app = FastAPI(
    title="AI Medical Receptionist Agent",
    description="Hospital AI powered by HuggingFace + FAISS + RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR  = os.path.join(BASE_DIR, "data", "csv")
PDF_DIR  = os.path.join(BASE_DIR, "data", "pdf")
TXT_DIR  = os.path.join(BASE_DIR, "data", "pdf")

# ── Initialize Agent (startup) ────────────────────────────────
agent         = None
whisper_model = None

@app.on_event("startup")
async def startup():
    global agent, whisper_model
    logger.info("⏳ Loading AI Agent...")
    agent = MedicalReceptionistAgent(CSV_DIR, PDF_DIR, TXT_DIR)
    logger.info("⏳ Loading Whisper...")
    # whisper_model = whisper.load_model("base")
    logger.info("✅ All systems ready!")

# ── Schemas ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    source: str
    intent: str
    type:   str

# ── Chat Endpoint ─────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not agent:
        raise HTTPException(503, "Agent not ready")
    result = agent.chat(req.message)
    return ChatResponse(**result)

# ── Voice Chat Endpoint ───────────────────────────────────────
@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    if not whisper_model or not agent:
        raise HTTPException(503, "Agent not ready")

    # Save uploaded audio
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Transcribe
    result   = whisper_model.transcribe(tmp_path)
    question = result["text"].strip()
    os.unlink(tmp_path)

    if not question:
        raise HTTPException(400, "Could not transcribe audio")

    # Get answer
    answer = agent.chat(question)
    return {**answer, "question": question}

# ── CSV Endpoints ─────────────────────────────────────────────
@app.get("/doctors")
async def get_doctors(specialization: str = None):
    import pandas as pd
    df = pd.read_csv(os.path.join(CSV_DIR, "doctors.csv"))
    if specialization:
        df = df[df["specialization"].str.lower().str.contains(specialization.lower(), na=False)]
    return df.to_dict(orient="records")

@app.get("/beds")
async def get_beds(ward: str = None):
    import pandas as pd
    df = pd.read_csv(os.path.join(CSV_DIR, "beds.csv"))
    if ward:
        df = df[df["ward_name"].str.lower().str.contains(ward.lower(), na=False)]
    return df.to_dict(orient="records")

@app.get("/departments")
async def get_departments():
    import pandas as pd
    df = pd.read_csv(os.path.join(CSV_DIR, "departments.csv"))
    return df.to_dict(orient="records")

@app.get("/pharmacy")
async def get_pharmacy():
    import pandas as pd
    df = pd.read_csv(os.path.join(CSV_DIR, "pharmacy.csv"))
    return df.to_dict(orient="records")

@app.get("/patients")
async def get_patients():
    import pandas as pd
    path = os.path.join(CSV_DIR, "patients.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/appointments")
async def get_appointments():
    import pandas as pd
    path = os.path.join(CSV_DIR, "appointments.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/billing")
async def get_billing():
    import pandas as pd
    path = os.path.join(CSV_DIR, "billing.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/staff")
async def get_staff():
    import pandas as pd
    path = os.path.join(CSV_DIR, "staff.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/health")
async def health():
    return {"status": "ok", "agent": "ready" if agent else "loading"}

# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=False
    )
