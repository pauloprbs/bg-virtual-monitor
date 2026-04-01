import os
import sys
import fitz
from sqlalchemy import text
from sqlalchemy.orm import Session

# ── Path e Imports do App ─────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from app.database.database import SessionLocal, engine
from app.database.models import Base, Game, GameChunk

# ── Configuração de NLP ───────────────────────────────────────────────────────
embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://ollama:11434"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

def process_manuals():
    # 1. Preparação do Banco de Dados
    # Garante que o Postgres está pronto com as tabelas necessárias antes de prosseguir
    print("🛠️ Preparando tabelas e extensões no banco de dados...")
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(bind=conn)

    db: Session = SessionLocal()
    manuals_dir = "data/manuals"

    # 2. Varredura de arquivos
    if not os.path.exists(manuals_dir):
        print(f"❌ Erro: Pasta '{manuals_dir}' não encontrada.")
        return

    files = [f for f in os.listdir(manuals_dir) if f.endswith(".pdf")]

    if not files:
        print("⚠️ Pasta 'data/manuals' está vazia ou não contém PDFs.")
        return

    for file_name in files:
        # Normaliza o título (ex: brass_birmingham.pdf -> Brass Birmingham)
        game_title = file_name.replace(".pdf", "").replace("_", " ").title()

        # 3. Verificação de Duplicidade
        existing_game = db.query(Game).filter(Game.title == game_title).first()
        if existing_game:
            print(f"⏩ Pulando '{game_title}' (já está no banco).")
            continue

        print(f"🚀 Iniciando ingestão de: {game_title}...")
        path = os.path.join(manuals_dir, file_name)

        try:
            # Criar registro do jogo
            new_game = Game(title=game_title)
            db.add(new_game)
            db.commit()
            db.refresh(new_game)

            # 4. Extração e Chunking
            doc = fitz.open(path)
            all_chunks = []

            for page_num, page in enumerate(doc):
                text_content = page.get_text()

                if not text_content.strip():
                    continue

                chunks = text_splitter.split_text(text_content)

                for content in chunks:
                    # Gera o embedding vetorial via Ollama (bge-m3)
                    vector = embeddings.embed_query(content)

                    all_chunks.append(
                        GameChunk(
                            game_id=new_game.id,
                            content=content,
                            embedding=vector,
                            page_number=page_num + 1
                        )
                    )

            if all_chunks:
                db.add_all(all_chunks)
                db.commit()
                print(f"✅ Concluído: {game_title} ({len(all_chunks)} fragmentos).")
            else:
                print(f"⚠️ Aviso: '{game_title}' não gerou fragmentos (PDF pode ser apenas imagem).")

        except Exception as e:
            db.rollback()
            print(f"❌ Erro ao processar '{file_name}': {e}")

    db.close()

if __name__ == "__main__":
    process_manuals()