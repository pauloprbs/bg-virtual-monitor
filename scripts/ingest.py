import os
import fitz
from sqlalchemy.orm import Session

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from app.database.database import SessionLocal
from app.database.models import Game, GameChunk

# Configuração
embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://ollama:11434"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

def process_manuals():
    db: Session = SessionLocal()
    manuals_dir = "data/manuals"

    # 1. Lista arquivos PDF na pasta
    files = [f for f in os.listdir(manuals_dir) if f.endswith(".pdf")]

    if not files:
        print("folder 'data/manuals' is empty or has no PDFs.")
        return

    for file_name in files:
        game_title = file_name.replace(".pdf", "").replace("_", " ").title()

        # 2. Verifica se já foi ingestado
        existing_game = db.query(Game).filter(Game.title == game_title).first()
        if existing_game:
            print(f"⏩ Pulando '{game_title}' (já está no banco).")
            continue

        print(f"🚀 Iniciando ingestão de: {game_title}...")
        path = os.path.join(manuals_dir, file_name)

        # Criar registro do jogo
        new_game = Game(title=game_title)
        db.add(new_game)
        db.commit()
        db.refresh(new_game)

        # 3. Processamento
        doc = fitz.open(path)
        all_chunks = []

        for page_num, page in enumerate(doc):
            text = page.get_text()

            if not text.strip():
                continue

            chunks = text_splitter.split_text(text)

            for content in chunks:
                vector = embeddings.embed_query(content)

                all_chunks.append(
                    GameChunk(
                        game_id=new_game.id,
                        content=content,
                        embedding=vector,
                        page_number=page_num + 1
                    )
                )

        db.add_all(all_chunks)
        db.commit()

        print(f"✅ Concluído: {game_title} ({len(all_chunks)} fragmentos).")

    db.close()


if __name__ == "__main__":
    process_manuals()