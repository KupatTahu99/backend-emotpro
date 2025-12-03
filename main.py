from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import text, multimodal, vision 

app = FastAPI(
    title="Emotion & Anger Detection API",
    description="Multimodal emotion detection system (Text + Face)",
    version="1.0.0"
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(text.router, prefix="/api", tags=["Text Analysis"])
app.include_router(multimodal.router, prefix="/api", tags=["Multimodal Analysis"])
app.include_router(vision.router) 


# --- HEALTH CHECK ---
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

# --- ENTRY POINT ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)