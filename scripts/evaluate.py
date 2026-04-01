import os
import sys
import json
import time
import unicodedata
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
from rank_bm25 import BM25Okapi

# ── Path e Imports do App ─────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from app.services.chat_service import get_answer
from app.utils.text_processor import super_clean

# ── Configurações ─────────────────────────────────────────────────────────────
GAME_TITLE   = "Brass Birmingham"
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
DATABASE_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST', 'db')}:{os.getenv('POSTGRES_PORT', '5432')}"
    f"/{os.getenv('POSTGRES_DB')}"
)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

if LLM_PROVIDER == "ollama":
    print(f"🤖 Avaliador Local: Ollama ({LLM_MODEL})")
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0, num_ctx=8192)
else:
    print(f"☁️ Avaliador Nuvem: Groq ({LLM_MODEL})")
    llm = ChatGroq(model_name=LLM_MODEL, groq_api_key=os.getenv("GROQ_API_KEY"),
                   temperature=0, max_tokens=4096)

llm_wrapper      = LangchainLLMWrapper(llm)
embeddings_model = OllamaEmbeddings(model="bge-m3", base_url=OLLAMA_URL)
emb_wrapper      = LangchainEmbeddingsWrapper(embeddings_model)

engine  = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)

# ── Golden Set ────────────────────────────────────────────────────────────────
QUESTIONS = [
    "Como se consegue a vitória no jogo?",
    "Quantos jogadores podem participar?",
    "O que posso fazer/executar no meu turno?",
    "Como termina uma rodada?",
    "Como e determinada a ordem dos turnos?",
    "Quais sao os componentes incluidos na caixa do jogo?",
    "Para que serve o tabuleiro principal?",
    "O que acontece quando um jogador nao pode executar uma acao por falta de recursos?",
    "Como a pontuacao final e calculada e quais elementos contribuem para ela?",
    "Quando posso passar meu turno e o que isso implica?",
    "Como deve ser feita a preparacao (setup) inicial dos componentes no tabuleiro?",
    "O que acontece quando algum componente limitado do jogo e exaurido?",
    "Existe alguma condicao que encerra o jogo antes do fim previsto?",
    "O manual oferece alguma dica ou orientacao estrategica para os jogadores?",
    "Existe alguma expansao oficial para este jogo?",
    "Quanto tempo dura uma partida em media?",
    "O que acontece se um jogador cometer um erro em uma acao ja executada?",
    "O que acontece se dois jogadores empatarem na ordem de turno?",
    "Posso executar a mesma acao mais de uma vez no mesmo turno?",
    "Quem e declarado vencedor em caso de empate na pontuacao final?",
]

EVIDENCE_MAP = {
    QUESTIONS[0]: ["vencedor", "vencer", "vitoria", "objetivo", "pontos de vitoria"],
    QUESTIONS[1]: ["jogadores", "participantes", "quantidade"],
    QUESTIONS[2]: ["turno", "acoes", "fases", "executar"],
    QUESTIONS[3]: ["rodada", "fim de rodada", "termina"],
    QUESTIONS[4]: ["ordem", "sequencia", "proximo", "trilha"],
    QUESTIONS[5]: ["componentes", "lista", "caixa", "incluidos"],
    QUESTIONS[6]: ["tabuleiro", "mapa", "principal", "espacos"],
    QUESTIONS[9]: ["passar", "encerrar", "terminar", "descartar"],
    QUESTIONS[10]: ["preparacao", "setup", "inicial", "configuracao"],
    QUESTIONS[11]: ["exaurido", "esgotar", "acabar", "limite", "vazio"],
    QUESTIONS[17]: ["empate", "iguais", "mesma quantidade", "ordem relativa"],
    QUESTIONS[18]: ["mesma", "repetir", "duas vezes", "novamente"],
}

# ── Utilitários ───────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    if not text: return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

def tokenize(text: str) -> list[str]:
    return normalize(text).split()

def load_all_chunks() -> list[tuple]:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT c.content, c.page_number
            FROM game_chunks c
            JOIN games g ON c.game_id = g.id
            WHERE g.title ILIKE :title
            ORDER BY c.page_number
        """), {"title": f"%{GAME_TITLE}%"}).fetchall()
    return [(r.content, r.page_number) for r in rows]

# ── Retrievers ────────────────────────────────────────────────────────────────
def retrieve_dense(question: str, k: int = 10) -> list[str]:
    vector = embeddings_model.embed_query(question)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT c.content FROM game_chunks c
            JOIN games g ON c.game_id = g.id
            WHERE g.title ILIKE :title
            ORDER BY c.embedding <=> :vector LIMIT :k
        """), {"vector": str(vector), "title": f"%{GAME_TITLE}%", "k": k}).fetchall()
    return [r.content for r in rows]

def retrieve_sparse(question: str, bm25: BM25Okapi, chunks: list[tuple], k: int = 10) -> list[str]:
    scores  = bm25.get_scores(tokenize(question))
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i][0] for i in indices]

def retrieve_hybrid(question: str, bm25: BM25Okapi, chunks: list[tuple], k: int = 10) -> list[str]:
    dense_results  = [(c, 0) for c in retrieve_dense(question, k=k)]
    sparse_scores  = bm25.get_scores(tokenize(question))
    sparse_indices = sorted(range(len(sparse_scores)), key=lambda i: sparse_scores[i], reverse=True)[:k]
    sparse_results = [(chunks[i][0], chunks[i][1]) for i in sparse_indices]

    RRF_K = 60
    scores_rrf: dict[str, float] = {}
    texts: dict[str, str] = {}
    for ranked_list in [dense_results, sparse_results]:
        for pos, (content, _) in enumerate(ranked_list):
            fp = " ".join(normalize(super_clean(content)).split()[:20])
            scores_rrf[fp] = scores_rrf.get(fp, 0.0) + 1.0 / (RRF_K + pos + 1)
            if fp not in texts: texts[fp] = content
    sorted_fps = sorted(scores_rrf, key=lambda fp: scores_rrf[fp], reverse=True)[:k]
    return [texts[fp] for fp in sorted_fps]

# ── Recall@k Trade-off ────────────────────────────────────────────────────────
def compute_recall_at_k(bm25: BM25Okapi, chunks: list[tuple]):
    print(f"\n=== Recall@k — {GAME_TITLE} (Evidence Sets) ===")
    print(f"{'Modo':<10} {'k=3':>8} {'k=5':>8} {'k=10':>8}")
    print("-" * 38)
    
    results = {}
    for mode in ("denso", "esparso", "hibrido"):
        row = []
        for k in (3, 5, 10):
            hits = 0
            for q, kw_list in EVIDENCE_MAP.items():
                # Garante que tratamos como lista mesmo que tenha apenas uma string
                if isinstance(kw_list, str):
                    kw_list = [kw_list]
                
                if mode == "denso": retrieved = retrieve_dense(q, k=k)
                elif mode == "esparso": retrieved = retrieve_sparse(q, bm25, chunks, k=k)
                else: retrieved = retrieve_hybrid(q, bm25, chunks, k=k)

                # Normaliza os chunks uma única vez para performance
                normalized_chunks = [normalize(super_clean(c)) for c in retrieved]
                
                # HIT se qualquer palavra-chave da lista estiver em qualquer chunk
                found = False
                for kw in kw_list:
                    kw_norm = normalize(kw)
                    if any(kw_norm in c for c in normalized_chunks):
                        found = True
                        break
                
                hits += int(found)
            
            recall = hits / len(EVIDENCE_MAP)
            row.append(f"{recall:.2f} ({hits}/{len(EVIDENCE_MAP)})")
            
        results[mode] = row
        print(f"{mode:<10} {row[0]:>8} {row[1]:>8} {row[2]:>8}")
    
    return results

# ── Geração do Dataset com Throttling Reforçado ───────────────────────────────
def build_ragas_dataset(bm25: BM25Okapi, chunks: list[tuple]) -> Dataset:
    db, records = Session(), []
    print(f"\n--- Gerando Corpus (20 Perguntas) ---")

    for i, q in enumerate(QUESTIONS):
        print(f"  [{i+1:02d}/20] {q[:50]}...")
        success = False
        retry_wait = 60 # Tempo inicial de espera caso dê erro 429

        while not success:
            try:
                # 1. Gera resposta (Pode bater no limite se o contexto for grande)
                answer, _ = get_answer(q, GAME_TITLE, db)
                
                # 2. Busca contextos híbridos
                ctxs = retrieve_hybrid(q, bm25, chunks, k=10)
                
                # 3. Gera Ground Truth
                gt_prompt = (
                    f"Responda concisamente usando apenas estes trechos do manual: '{q}'\n\n"
                    "TRECHOS:\n" + "\n\n".join(ctxs[:4])
                )
                ground_truth = llm.invoke(gt_prompt).content
                
                records.append({"question": q, "answer": answer, "contexts": ctxs, "ground_truth": ground_truth})
                
                # Pausa preventiva obrigatória para o Groq
                if LLM_PROVIDER == "groq":
                    time.sleep(45) 
                
                success = True
            except Exception as e:
                if "429" in str(e):
                    print(f"  ⏳ Rate limit atingido. Aguardando {retry_wait}s...")
                    time.sleep(retry_wait)
                    retry_wait *= 1.5 # Backoff exponencial se persistir
                else:
                    print(f"  ❌ Erro fatal na questão {i+1}: {e}")
                    success = True # Sai do loop para não travar o script

    db.close()
    return Dataset.from_list(records)

# ── Execução RAGAS com Throttling ─────────────────────────────────────────────
def run_ragas_in_batches(dataset: Dataset, batch_size: int = 5):
    print(f"\n--- Avaliação RAGAS (Lotes de {batch_size}) ---")
    # Configuração de workers=1 é CRÍTICA para evitar 429 simultâneos
    config = RunConfig(max_workers=1, timeout=900, max_retries=10)
    all_dfs = []
    
    for i in range(0, len(dataset), batch_size):
        end = min(i + batch_size, len(dataset))
        print(f"  Processando {i+1} a {end}...")
        batch = dataset.select(range(i, end))
        
        success = False
        while not success:
            try:
                res = evaluate(batch, metrics=[faithfulness], llm=llm_wrapper, embeddings=emb_wrapper, run_config=config)
                all_dfs.append(res.to_pandas())
                
                if LLM_PROVIDER == "groq":
                    time.sleep(40) # Descanso entre lotes
                success = True
            except Exception as e:
                if "429" in str(e):
                    print("  ⏳ Rate limit no RAGAS. Pausando 60s...")
                    time.sleep(60)
                else:
                    print(f"  ⚠️ Erro no processamento: {e}")
                    success = True

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        output = f"ragas_{GAME_TITLE.lower().replace(' ','_')}_{LLM_PROVIDER}.json"
        final_df.to_json(output, orient="records", indent=2, force_ascii=False)
        print(f"\n✅ Faithfulness Média: {final_df['faithfulness'].mean():.4f}")
        return final_df
    return None

if __name__ == "__main__":
    all_chunks = load_all_chunks()
    bm25_index = BM25Okapi([tokenize(super_clean(c)) for c, _ in all_chunks])
    
    compute_recall_at_k(bm25_index, all_chunks)
    ds = build_ragas_dataset(bm25_index, all_chunks)
    if ds: run_ragas_in_batches(ds)