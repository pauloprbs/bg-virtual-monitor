from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from .database import Base

class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, unique=True, index=True, nullable=False)
    publisher = Column(String)
    
    # Relacionamento: Um jogo tem muitos pedaços (chunks) de regras
    chunks = relationship("GameChunk", back_populates="game", cascade="all, delete-orphan")

class GameChunk(Base):
    __tablename__ = "game_chunks"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    content = Column(Text, nullable=False)  # O texto da regra em si
    
    # O BGE-M3 gera 1024 dimensões. 
    # Essa coluna armazena a "posição" semântica do texto.
    embedding = Column(Vector(1024)) 
    
    page_number = Column(Integer)
    
    game = relationship("Game", back_populates="chunks")