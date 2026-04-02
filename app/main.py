from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Literal

from app.database.database import engine, Base, get_db
from app.database import models
from app.services.chat_service import get_answer


class ChatRequest(BaseModel):
    game_title: str
    question: str
    mode: Literal["denso", "esparso", "hibrido"] = "hibrido"


@asynccontextmanager
async def lifespan(app: FastAPI):
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(bind=conn)
    yield
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
    Pergunta sobre as regras de um jogo.
    - game_title: ex. 'Brass Birmingham'
    - mode: 'denso' | 'esparso' | 'hibrido' (padrão: hibrido)
    """
    try:
        answer, sources = get_answer(
            request.question,
            request.game_title,
            db,
            mode=request.mode
        )
        return {
            "game": request.game_title,
            "question": request.question,
            "mode": request.mode,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        print(f"Erro no processamento do chat: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao processar a pergunta.")


@app.get("/games")
async def list_games(db: Session = Depends(get_db)):
    games = db.query(models.Game.title).all()
    return [g.title for g in games]