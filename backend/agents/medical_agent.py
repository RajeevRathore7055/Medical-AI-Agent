import os
import logging
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from backend.rag.rag_engine import RAGEngine
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class MedicalReceptionistAgent:
    def __init__(self, csv_dir: str, pdf_dir: str, txt_dir: str):
        self.csv_dir = csv_dir
        self.pdf_dir = pdf_dir
        self.txt_dir = txt_dir
        self.data    = {}
        self.rag     = RAGEngine(pdf_dir, txt_dir)
        self._load_model()
        self._load_csv_data()
        self.rag.build_index()
        logger.info("✅ Medical Agent ready!")

    # ── Load HuggingFace LLM ──────────────────────────────────
    def _load_model(self):
        model_name = os.getenv("LLM_MODEL", "google/flan-t5-base")
        logger.info(f"⏳ Loading LLM: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model     = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        logger.info("✅ LLM ready!")

    # ── Load CSV Data ─────────────────────────────────────────
    def _load_csv_data(self):
        csv_files = [
            "doctors", "patients", "appointments",
            "billing", "beds", "departments",
            "staff", "pharmacy", "diagnostics"
        ]
        for name in csv_files:
            path = os.path.join(self.csv_dir, f"{name}.csv")
            if os.path.exists(path):
                self.data[name] = pd.read_csv(path)
                logger.info(f"✅ Loaded {name}.csv: {len(self.data[name])} records")

    # ── LLM Generate ─────────────────────────────────────────
    def _llm_generate(self, prompt: str, max_length: int = 300) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=2,
                early_stopping=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ── Intent Detection ──────────────────────────────────────
    def _detect_intent(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["doctor", "physician", "specialist", "surgeon", "cardiolog", "orthoped", "neurolog", "gynecol", "pediatr", "dermatol", "oncolog", "ent", "ophthal"]):
            return "doctors"
        if any(w in q for w in ["bed", "ward", "icu", "available", "occupied", "room"]):
            return "beds"
        if any(w in q for w in ["appointment", "schedule", "booking", "slot"]):
            return "appointments"
        if any(w in q for w in ["patient", "admitted", "discharge"]):
            return "patients"
        if any(w in q for w in ["bill", "billing", "payment", "fee", "cost", "charges", "amount"]):
            return "billing"
        if any(w in q for w in ["department", "unit", "section"]):
            return "departments"
        if any(w in q for w in ["staff", "nurse", "nurse", "receptionist", "employee"]):
            return "staff"
        if any(w in q for w in ["medicine", "pharmacy", "drug", "tablet", "capsule", "stock"]):
            return "pharmacy"
        if any(w in q for w in ["test", "report", "lab", "diagnostic", "xray", "scan", "mri", "blood"]):
            return "diagnostics"
        if any(w in q for w in ["rule", "regulation", "policy", "smoking", "visiting hour", "emergency", "contact"]):
            return "pdf"
        return "general"

    # ── Format CSV Answer ─────────────────────────────────────
    def _answer_from_csv(self, intent: str, query: str) -> tuple[str, str]:
        q  = query.lower()
        df = self.data.get(intent)

        if df is None:
            return "Data not available.", "N/A"

        source = f"{intent}.csv"

        # DOCTORS
        if intent == "doctors":
            if "name" not in df.columns and "first_name" in df.columns and "last_name" in df.columns:
                df["name"] = df["first_name"] + " " + df["last_name"]
            
            display_cols = [c for c in ["name", "specialization", "timing", "room_number", "hospital_branch", "contact", "phone_number"] if c in df.columns]

            for spec in ["cardiolog", "orthoped", "neurolog", "gynecol", "pediatr", "dermatol", "oncolog", "ent", "ophthal", "general"]:
                if spec in q:
                    filtered = df[df["specialization"].str.lower().str.contains(spec, na=False)]
                    if not filtered.empty:
                        rows = filtered[display_cols].to_string(index=False)
                        return f"Available {spec.capitalize()} Doctors:\n\n{rows}", source

            # Today's available doctors
            if "today" in q or "available" in q:
                if "available_days" in df.columns:
                    import datetime
                    day = datetime.datetime.now().strftime("%a")
                    filtered = df[df["available_days"].str.contains(day, na=False)]
                    rows = filtered[display_cols].to_string(index=False)
                    return f"Doctors available today ({day}):\n\n{rows}", source
                else:
                    rows = df[display_cols].to_string(index=False)
                    return f"All Doctors (Schedule not available):\n\n{rows}", source

            rows = df[display_cols].to_string(index=False)
            return f"All Doctors:\n\n{rows}", source

        # BEDS
        if intent == "beds":
            status_col = "status" if "status" in df.columns else "is_available"
            ward_col = "ward_name" if "ward_name" in df.columns else ("ward_id" if "ward_id" in df.columns else None)
            
            total     = len(df)
            if status_col in df.columns:
                if status_col == "status":
                    available = len(df[df[status_col] == "Available"])
                else:
                    available = len(df[df[status_col] == True])
            else:
                available = 0
            
            occupied  = total - available
            summary   = f"Bed Summary:\n  Total: {total} | Available: {available} | Occupied: {occupied}\n\n"
            
            if ward_col and ward_col in df.columns:
                if status_col == "status":
                    ward_summary = df.groupby(ward_col).apply(
                        lambda x: f"  {x.name}: {len(x[x[status_col] == 'Available'])} available / {len(x)} total"
                    ).to_string()
                else:
                    ward_summary = df.groupby(ward_col).apply(
                        lambda x: f"  {x.name}: {x[status_col].sum() if status_col in df.columns else 0} available / {len(x)} total"
                    ).to_string()
                return summary + "Ward-wise Beds:\n" + ward_summary, source
            return summary, source

        # APPOINTMENTS
        if intent == "appointments":
            cols = [c for c in ["patient_name", "patient_id", "doctor_name", "doctor_id", "appointment_date", "appointment_time", "status"] if c in df.columns]
            rows = df[cols].head(10).to_string(index=False) if cols else "No appointment details available."
            return f"Recent Appointments:\n\n{rows}", source

        # PATIENTS
        if intent == "patients":
            if "name" not in df.columns and "first_name" in df.columns and "last_name" in df.columns:
                df["name"] = df["first_name"] + " " + df["last_name"]
            cols = [c for c in ["name", "age", "date_of_birth", "ward", "bed_number", "doctor_name", "diagnosis", "status", "contact_number"] if c in df.columns]
            rows = df[cols].head(10).to_string(index=False) if cols else "No patient details available."
            return f"Patient List:\n\n{rows}", source

        # BILLING
        if intent == "billing":
            cols = [c for c in ["patient_name", "bill_id", "total_amount", "paid_amount", "pending_amount", "patient_payable_amount", "payment_status"] if c in df.columns]
            rows = df[cols].head(10).to_string(index=False) if cols else "No billing details available."
            return f"Billing Details:\n\n{rows}", source

        # DEPARTMENTS
        if intent == "departments":
            cols = [c for c in ["name", "department_name", "head_doctor", "floor", "contact", "phone_extension", "total_beds"] if c in df.columns]
            rows = df[cols].to_string(index=False) if cols else "No department details available."
            return f"Hospital Departments:\n\n{rows}", source

        # STAFF
        if intent == "staff":
            cols = [c for c in ["name", "role", "department", "shift", "contact"] if c in df.columns]
            rows = df[cols].head(10).to_string(index=False) if cols else "No staff details available."
            return f"Staff List:\n\n{rows}", source

        # PHARMACY
        if intent == "pharmacy":
            cols = [c for c in ["medicine_name", "name", "category", "stock_quantity", "stock_qty", "price", "unit_price"] if c in df.columns]
            rows = df[cols].head(10).to_string(index=False) if cols else "No pharmacy details available."
            return f"Pharmacy Inventory:\n\n{rows}", source

        return "No data found.", source
    # ── Main Chat Function ────────────────────────────────────
    def chat(self, query: str) -> dict:
        intent = self._detect_intent(query)
        logger.info(f"Intent: {intent} | Query: {query}")

        # PDF/Knowledge Base queries
        if intent == "pdf" or intent == "general":
            results       = self.rag.search(query, top_k=3)
            context, src  = self.rag.format_context(results)

            if context:
                prompt = f"""
You are a helpful hospital receptionist.
Answer the question using this hospital information:

{context}

Question: {query}
Answer:"""
                answer = self._llm_generate(prompt)
                return {
                    "answer": answer,
                    "source": src,
                    "intent": intent,
                    "type":   "pdf"
                }

        # CSV queries
        if intent in self.data:
            answer, source = self._answer_from_csv(intent, query)
            return {
                "answer": answer,
                "source": source,
                "intent": intent,
                "type":   "csv"
            }

        # Fallback - LLM general answer
        prompt = f"You are a hospital receptionist. Answer this query: {query}"
        answer = self._llm_generate(prompt)
        return {
            "answer": answer,
            "source": "General Knowledge",
            "intent": intent,
            "type":   "llm"
        }
