import os
from sqlalchemy import text
from langchain_ollama import OllamaEmbeddings
from app.database.database import SessionLocal

embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://ollama:11434")

def search(question: str):
    db = SessionLocal()
    # 1. Transforma a pergunta em vetor
    print(f"🔎 Buscando por: '{question}'")
    query_vector = embeddings.embed_query(question)
    
    # 2. Busca no Postgres por proximidade (Cosine Similarity)
    # O operador <=> retorna a distância. Quanto menor, mais perto.
    sql = text("""
        SELECT content, page_number 
        FROM game_chunks 
        ORDER BY embedding <=> :vector 
        LIMIT 2
    """)
    
    results = db.execute(sql, {"vector": str(query_vector)}).fetchall()
    
    for i, res in enumerate(results):
        print(f"\nResultado {i+1} (Pág {res.page_number}):")
        print(f"{res.content[:300]}...") # Mostra só o começo do trecho
    
    db.close()

if __name__ == "__main__":
    # Teste com algo específico do Brass
    search("Como funciona a rodada?")