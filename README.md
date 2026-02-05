#MediAssist
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
from PIL import Image
import uuid
import os
import sqlite3

app = FastAPI(title="MediAssist AI Backend", version="0.2.0")

# ✅ Allow React dev server to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React (CRA)
        "http://localhost:5173",  # Vite
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DB_PATH = BASE_DIR / "mediassist.db"


# -------------------------
# Database helpers
# -------------------------
def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            request_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            symptoms TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            symptom_summary TEXT NOT NULL,
            image_observation TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            stored_image_path TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


@app.on_event("startup")
def on_startup():
    init_db()


# -------------------------
# Models
# -------------------------
class AnalyzeResponse(BaseModel):
    request_id: str
    timestamp: str
    risk_level: str
    symptom_summary: str
    image_observation: str
    recommendation: str
    stored_image_path: str


class HistoryItem(BaseModel):
    request_id: str
    timestamp: str
    risk_level: str
    symptom_summary: str
    image_observation: str


# -------------------------
# Utility logic (prototype)
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


def risk_from_symptoms(symptoms: str) -> tuple[str, str, str]:
    """
    Simple prototype scoring (rule-based). NOT diagnosis.
    """
    s = (symptoms or "").lower()
    s = symptoms.lower().replace(".", "").replace(",", "")


    red_flags = [
    "chest pain",
    "shortness of breath",
    "hard to breathe",
    "hard to breath",
    "breathing problem",
    "breathing difficulty",
    "difficulty breathing",
    "cannot breathe",
    "cant breathe",
    "breathless",
    "faint",
    "unconscious",
    "severe bleeding",
    "stroke",
    "seizure"
]
    medium_flags = [
        "fever", "cough", "vomit", "vomiting", "diarrhea",
        "headache", "dizziness", "infection", "pain"
    ]

    if any(k in s for k in red_flags):
        return (
            "High",
            "Red-flag symptoms detected (needs urgent attention).",
            "Seek urgent medical review / emergency evaluation."
        )
    if any(k in s for k in medium_flags):
        return (
            "Medium",
            "Common symptoms detected (moderate risk).",
            "Doctor review recommended if symptoms persist/worsen."
        )
    return (
        "Low",
        "No major red-flag indicators detected from symptom text.",
        "Self-care guidance; consult a doctor if unsure."
    )


def basic_image_observation(image_path: Path) -> str:
    """
    Prototype-only image check. Replace later with CNN inference.
    """
    try:
        with Image.open(image_path) as im:
            w, h = im.size
            mode = im.mode
        return f"Image received successfully ({w}x{h}, {mode}). AI imaging module ready for integration."
    except Exception:
        return "Image received, but could not read image details. (Supported formats: JPG/PNG)."


def save_analysis_to_db(resp: AnalyzeResponse, symptoms: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO analyses (
            request_id, timestamp, symptoms, risk_level, symptom_summary,
            image_observation, recommendation, stored_image_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            resp.request_id,
            resp.timestamp,
            symptoms,
            resp.risk_level,
            resp.symptom_summary,
            resp.image_observation,
            resp.recommendation,
            resp.stored_image_path,
        ),
    )
    conn.commit()
    conn.close()


# -------------------------
# Main endpoint
# -------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    symptoms: str = Form(...),
    image: UploadFile = File(...)
):
    filename = image.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG/PNG images are supported for this prototype.")

    request_id = str(uuid.uuid4())
    saved_name = f"{request_id}{ext}"
    saved_path = UPLOAD_DIR / saved_name

    content = await image.read()
    saved_path.write_bytes(content)

    risk_level, symptom_summary, rec = risk_from_symptoms(symptoms)
    img_obs = basic_image_observation(saved_path)

    now = datetime.utcnow().isoformat() + "Z"
    resp = AnalyzeResponse(
        request_id=request_id,
        timestamp=now,
        risk_level=risk_level,
        symptom_summary=symptom_summary,
        image_observation=img_obs,
        recommendation=rec,
        stored_image_path=str(saved_path).replace("\\", "/"),
    )

    # ✅ Save to SQLite
    save_analysis_to_db(resp, symptoms)

    return resp


# -------------------------
# History endpoints
# -------------------------
@app.get("/history", response_model=list[HistoryItem])
def history(limit: int = 20):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT request_id, timestamp, risk_level, symptom_summary, image_observation
        FROM analyses
        ORDER BY timestamp DESC
        LIMIT ?;
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()

    return [HistoryItem(**dict(r)) for r in rows]


@app.get("/history/{request_id}", response_model=AnalyzeResponse)
def history_by_id(request_id: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM analyses WHERE request_id = ?;", (request_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Request ID not found.")

    # Note: symptoms is stored too, but we return the same response model as /analyze
    data = dict(row)
    return AnalyzeResponse(
        request_id=data["request_id"],
        timestamp=data["timestamp"],
        risk_level=data["risk_level"],
        symptom_summary=data["symptom_summary"],
        image_observation=data["image_observation"],
        recommendation=data["recommendation"],
        stored_image_path=data["stored_image_path"],
    )# reimagined-lamp
MediAssist AI is a web-based HealthTech decision support platform that combines symptom analysis and medical image analysis into a single workflow. It provides preliminary risk assessment and clinical insights while keeping doctors in the decision loop, improving efficiency and reducing unnecessary hospital visits for minor health concerns.
