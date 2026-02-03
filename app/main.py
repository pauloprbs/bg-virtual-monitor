from fastapi import FastAPI

app = FastAPI(
    title="BG Virtual Monitor",
    description="O seu monitor virtual de jogos de tabuleiro modernos.",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {"status": "online", "message": "BG Virtual Monitor pronto para ensinar regras!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}