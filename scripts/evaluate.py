import os
import sys
import json
import time
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from datasets import Dataset

# ── Path ──────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from app.services.chat_service import get_answer
from app.utils.text_processor import super_clean

# ── Configuracoes ─────────────────────────────────────────────────────────────
GAME_TITLE   = "Brass Birmingham"
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
DATABASE_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST', 'db')}:{os.getenv('POSTGRES_PORT', '5432')}"
    f"/{os.getenv('POSTGRES_DB')}"
)

# ── Selecao de LLM (mesma logica do chat_service) ─────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

if LLM_PROVIDER == "ollama":
    print(f"LLM: Ollama ({LLM_MODEL})")
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0)
else:
    print(f"LLM: Groq ({LLM_MODEL})")
    llm = ChatGroq(
        model_name=LLM_MODEL,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        max_tokens=4096,
    )

# ── Banco e Embeddings ────────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("Conexao com o banco estabelecida!")
except Exception as e:
    print(f"ERRO DE CONEXAO: {e}")
    sys.exit(1)

Session    = sessionmaker(bind=engine)
embeddings = OllamaEmbeddings(model="bge-m3", base_url=OLLAMA_URL)

# ── Golden Set ────────────────────────────────────────────────────────────────
QUESTIONS = [
    "Qual e o objetivo do jogo?",
    "Quantos jogadores podem participar?",
    "Quais acoes posso executar no meu turno?",
    "Como termina uma rodada?",
    "Como e determinada a ordem dos turnos?",
    "Quais sao os componentes incluidos na caixa do jogo?",
    "Para que serve o tabuleiro principal?",
    "O que acontece quando um jogador nao pode executar uma acao por falta de recursos?",
    "Como a pontuacao final e calculada e quais elementos contribuem para ela?",
    "Quando posso passar meu turno e o que isso implica?",
    "Existe alguma regra especial que se aplica apenas em um momento especifico do jogo?",
    "O que acontece quando algum componente limitado do jogo se esgota?",
    "Existe alguma condicao que encerra o jogo antes do fim previsto?",
    "O manual oferece alguma dica ou orientacao estrategica para os jogadores?",
    "Existe alguma expansao oficial para este jogo?",
    "Quanto tempo dura uma partida em media?",
    "O que acontece se um jogador cometer um erro em uma acao ja executada?",
    "O que acontece se dois jogadores empatarem na ordem de turno?",
    "Posso executar a mesma acao mais de uma vez no mesmo turno?",
    "Quem e declarado vencedor em caso de empate na pontuacao final?",
]

QUESTIONS_WITH_EVIDENCE = [
    (QUESTIONS[0],  "objetivo"),
    (QUESTIONS[1],  "jogadores"),
    (QUESTIONS[2],  "acoes"),
    (QUESTIONS[3],  "rodada"),
    (QUESTIONS[4],  "ordem"),
    (QUESTIONS[5],  "componentes"),
    (QUESTIONS[6],  "tabuleiro"),
    (QUESTIONS[9],  "passar"),
    (QUESTIONS[10], "especial"),
    (QUESTIONS[11], "esgota"),
    (QUESTIONS[17], "empate"),
    (QUESTIONS[18], "mesma"),
]

# ── Funcoes ───────────────────────────────────────────────────────────────────
def retrieve_chunks(question, k=6):
    vector = embeddings.embed_query(question)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT c.content FROM game_chunks c
            JOIN games g ON c.game_id = g.id
            WHERE g.title ILIKE :title
            ORDER BY c.embedding <=> :vector LIMIT :k
        """), {"vector": str(vector), "title": f"%{GAME_TITLE}%", "k": k}).fetchall()
    return [super_clean(r.content) for r in rows]


def generate_ground_truth(question, chunks):
    context = "\n\n".join(chunks)
    prompt = (
        f"Com base nos trechos abaixo do manual, escreva a melhor resposta possivel "
        f"para: '{question}'\n"
        f"IMPORTANTE: Extraia qualquer informacao relevante dos trechos, mesmo que incompleta. "
        f"Somente responda 'Nao encontrei essa informacao no manual' se os trechos forem "
        f"completamente irrelevantes para a pergunta.\n\nTRECHOS:\n{context}"
    )
    return llm.invoke(prompt).content


def compute_recall_at_k():
    print("\n=== Recall@k ===")
    for k in (3, 5, 10):
        hits = 0
        for q, kw in QUESTIONS_WITH_EVIDENCE:
            chunks = retrieve_chunks(q, k=k)
            hit = any(kw.lower() in c.lower() for c in chunks)
            if not hit:
                print(f"  MISS (k={k}): '{kw}' nao encontrado para: {q[:50]}")
            hits += int(hit)
        print(f"Recall@{k}: {hits/len(QUESTIONS_WITH_EVIDENCE):.2f} ({hits}/{len(QUESTIONS_WITH_EVIDENCE)})")


def build_ragas_dataset():
    db, records = Session(), []
    print("\n--- Gerando Corpus (20 Perguntas) ---")
    for i, q in enumerate(QUESTIONS):
        print(f"  [{i+1:02d}/20] {q[:50]}...")
        answer, _ = get_answer(q, GAME_TITLE, db)
        ctxs      = retrieve_chunks(q, k=6)
        gt        = generate_ground_truth(q, ctxs)
        records.append({"question": q, "answer": answer, "contexts": ctxs, "ground_truth": gt})
        if LLM_PROVIDER != "ollama":
            time.sleep(0.5)  # rate limit so necessario no Groq
    db.close()
    return Dataset.from_list(records)


def run_ragas_in_batches(dataset, batch_size=5):
    print(f"\n--- Iniciando Avaliacao RAGAS (Lotes de {batch_size}) ---")
    llm_wrapper = LangchainLLMWrapper(llm)
    emb_wrapper = LangchainEmbeddingsWrapper(embeddings)
    config      = RunConfig(max_workers=1, timeout=240, max_retries=5)

    all_dfs = []
    for i in range(0, len(dataset), batch_size):
        end   = min(i + batch_size, len(dataset))
        print(f"  Avaliando perguntas {i+1} a {end}...")
        batch = dataset.select(range(i, end))
        try:
            res = evaluate(
                batch,
                metrics=[faithfulness],
                llm=llm_wrapper,
                embeddings=emb_wrapper,
                run_config=config,
            )
            all_dfs.append(res.to_pandas())
        except Exception as e:
            print(f"  Erro no lote: {e}")

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        output   = f"ragas_results_{LLM_PROVIDER}.json"
        final_df.to_json(output, orient="records", indent=2, force_ascii=False)
        print(f"\nAvaliacao concluida! Media de Faithfulness: {final_df['faithfulness'].mean():.4f}")
        print(f"Resultados salvos em {output}")
        return final_df
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    compute_recall_at_k()
    ds = build_ragas_dataset()
    run_ragas_in_batches(ds)