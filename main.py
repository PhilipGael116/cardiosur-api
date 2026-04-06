from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.predict import router as predict_router
from services.model import load_model

app = FastAPI(
    title="CardioSur API",
    description="AI-powered phonocardiography backend",
    version="1.0.0"
)

# Allow requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once when server starts
@app.on_event("startup")
def startup_event():
    load_model("model/arrythmia.tflite")

# Routes
app.include_router(predict_router, prefix="/api")

@app.get("/")
def health_check():
    return { "status": "CardioSur API is running" }