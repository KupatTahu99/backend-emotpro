import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import text, multimodal, vision 

app = FastAPI(
    title="Emotion & Anger Detection API",
    description="Multimodal emotion detection system (Text + Face)",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INCLUDE ROUTERS ---

# 1. Text Analysis
app.include_router(text.router, prefix="/api/text", tags=["Text Analysis"])

# 2. Multimodal Analysis
app.include_router(multimodal.router, prefix="/api/multimodal", tags=["Multimodal Analysis"])

# 3. Vision / Face Analysis
app.include_router(vision.router) 


# --- HEALTH CHECK ---
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.get("/")
async def root():
    return {"message": "Welcome to Emotion Detection API. Documentation at /docs"}

# --- ENTRY POINT (Untuk Deployment) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)