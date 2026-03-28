from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database.database import engine, Base, get_db
from app.database import models 
from app.services.chat_service import get_answer

# Modelo para a requisição de chat
class ChatRequest(BaseModel):
    game_title: str
    question: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Lógica de Inicialização (Startup) ---
    with engine.begin() as conn:
        # 1. Garante a extensão para vetores no Postgres
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # 2. Cria as tabelas baseadas nos modelos SQLAlchemy
        Base.metadata.create_all(bind=conn)
    
    yield  # A API fica online aqui
    
    # --- Lógica de Encerramento (Shutdown) ---
    engine.dispose()

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

@app.post("/ask")
async def ask_question(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Endpoint para perguntar sobre as regras de um jogo específico.
    Exemplo de game_title: 'Brass Birmingham' ou 'Catan'
    """
    try:
        answer, sources = get_answer(request.question, request.game_title, db)
        return {
            "game": request.game_title,
            "question": request.question,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        # Log do erro para facilitar o debug se o Groq ou Ollama falharem
        print(f"Erro no processamento do chat: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao processar a pergunta.")
    
@app.get("/games")
async def list_games(db: Session = Depends(get_db)):
    # Busca todos os títulos de jogos cadastrados
    games = db.query(models.Game.title).all()
    # Retorna apenas uma lista de strings: ["Brass Birmingham", "Catan"]
    return [g.title for g in games]