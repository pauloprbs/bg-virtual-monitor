from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import text
from app.database.database import engine, Base
from app.database import models 

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Lógica de Inicialização (Startup) ---
    with engine.begin() as conn:
        # 1. Garante a extensão para vetores no Postgres
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # 2. Cria as tabelas baseadas nos modelos SQLAlchemy
        Base.metadata.create_all(bind=conn)
    
    yield  # Aqui a API "roda". O código depois do yield só executa no shutdown.
    
    # --- Lógica de Encerramento (Shutdown) se necessário ---
    # engine.dispose() # Exemplo

app = FastAPI(
    title="BG Virtual Monitor",
    description="O seu monitor virtual de jogos de tabuleiro modernos.",
    version="0.1.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"status": "online", "message": "BG Virtual Monitor pronto para ensinar regras!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}