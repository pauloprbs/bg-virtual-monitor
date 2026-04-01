import os
import json
import unicodedata
from sqlalchemy.orm import Session
from sqlalchemy import text
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi

from app.utils.text_processor import super_clean

# ── Embeddings ────────────────────────────────────────────────────────────────
embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://ollama:11434")

# ── Seleção de LLM via variável de ambiente ───────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

if LLM_PROVIDER == "ollama":
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0)
else:
    llm = ChatGroq(
        temperature=0,
        model_name=LLM_MODEL,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

# ── Constante do RRF ──────────────────────────────────────────────────────────
RRF_K = 60


# ── Normalização para BM25 ────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    """Remove acentos e lowercaseapara tokenização BM25 consistente."""
    nfkd = unicodedata.normalize("NFKD", text)
    sem_acento = "".join(c for c in nfkd if not unicodedata.combining(c))
    return sem_acento.lower()


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


# ── Cache BM25 por jogo ───────────────────────────────────────────────────────
# Evita reindexar a cada chamada — o índice é construído uma vez por sessão.
_bm25_cache: dict[str, tuple] = {}  # game_title -> (BM25Okapi, [(content, page_number)])


def _get_bm25_index(game_title: str, db: Session) -> tuple:
    """Carrega todos os chunks do jogo e constrói (ou retorna do cache) o índice BM25."""
    if game_title in _bm25_cache:
        return _bm25_cache[game_title]

    sql = text("""
        SELECT c.content, c.page_number
        FROM game_chunks c
        JOIN games g ON c.game_id = g.id
        WHERE g.title ILIKE :title
        ORDER BY c.page_number
    """)
    rows = db.execute(sql, {"title": f"%{game_title}%"}).fetchall()
    chunks = [(r.content, r.page_number) for r in rows]

    corpus_tokens = [_tokenize(super_clean(content)) for content, _ in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    _bm25_cache[game_title] = (bm25, chunks)
    return bm25, chunks


# ── Retriever Esparso (BM25) ──────────────────────────────────────────────────
def search_chunks_bm25(query: str, game_title: str, db: Session, limit: int = 6) -> list[tuple]:
    """Busca esparsa por palavra-chave — retorna lista de (content, page_number)."""
    bm25, chunks = _get_bm25_index(game_title, db)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Pega os top-limit índices ordenados por score decrescente
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
    return [chunks[i] for i in ranked_indices]


# ── Retriever Denso (vetorial) ────────────────────────────────────────────────
def search_chunks_dense(query: str, game_title: str, db: Session, limit: int = 6) -> list[tuple]:
    """Busca densa via embedding — retorna lista de (content, page_number)."""
    query_vector = embeddings.embed_query(query)
    sql = text("""
        SELECT c.content, c.page_number
        FROM game_chunks c
        JOIN games g ON c.game_id = g.id
        WHERE g.title ILIKE :title
        ORDER BY c.embedding <=> :vector
        LIMIT :limit
    """)
    rows = db.execute(sql, {
        "vector": str(query_vector),
        "title": f"%{game_title}%",
        "limit": limit
    }).fetchall()
    return [(r.content, r.page_number) for r in rows]


# ── RRF Fusion ────────────────────────────────────────────────────────────────
def rrf_fusion(ranked_lists: list[list[tuple]], k: int = RRF_K) -> list[tuple]:
    """
    Reciprocal Rank Fusion (Cormack et al., 2009).
    Recebe N listas ordenadas de (content, page_number).
    Retorna lista fundida por score RRF acumulado, deduplicada por fingerprint.
    """
    scores: dict[str, float] = {}
    chunk_data: dict[str, tuple] = {}

    for ranked_list in ranked_lists:
        for position, (content, page_number) in enumerate(ranked_list):
            texto_limpo = super_clean(content)
            fingerprint = " ".join(texto_limpo.split()[:20])
            scores[fingerprint] = scores.get(fingerprint, 0.0) + 1.0 / (k + position + 1)
            if fingerprint not in chunk_data:
                chunk_data[fingerprint] = (page_number, texto_limpo)

    sorted_fps = sorted(scores, key=lambda fp: scores[fp], reverse=True)
    return [chunk_data[fp] for fp in sorted_fps]


# ── Query Expansion ───────────────────────────────────────────────────────────
def expand_query(question: str, game_title: str) -> list[str]:
    """Gera 2 variações da pergunta para ampliar o recall do retriever denso."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Você é um assistente especialista em Board Games que ajuda a melhorar buscas "
         "no manual de {game_title}. Dado uma pergunta de um jogador, gere EXATAMENTE 2 "
         "variações usando termos técnicos e vocabulário específico que costuma aparecer "
         "em manuais de regras (ex: setup, era, turno, acoes). "
         "Responda APENAS com um JSON array de strings. "
         "Exemplo: [\"variacao 1\", \"variacao 2\"]"),
        ("human", "{question}")
    ])
    chain = prompt | llm
    response = chain.invoke({"question": question, "game_title": game_title})
    try:
        variations = json.loads(response.content)
        if isinstance(variations, list):
            return [str(v) for v in variations[:2]]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


# ── Pipeline Principal ────────────────────────────────────────────────────────
def get_answer(question: str, game_title: str, db: Session):
    # 1. Busca densa original
    dense_original = search_chunks_dense(question, game_title, db, limit=10)

    if not dense_original:
        return "Lamentavelmente, não encontrei informações sobre este jogo no manual.", []

    # 2. Query expansion — 2 variações densas adicionais
    variations = expand_query(question, game_title)
    all_ranked_lists = [dense_original]
    for variation in variations:
        variation_results = search_chunks_dense(variation, game_title, db, limit=10)
        if variation_results:
            all_ranked_lists.append(variation_results)

    # 3. Busca esparsa BM25 — recuperação por palavra-chave exata
    sparse_results = search_chunks_bm25(question, game_title, db, limit=10)
    if sparse_results:
        all_ranked_lists.append(sparse_results)

    # 4. RRF híbrido — funde denso + expansão + esparso, pega top-10
    fused_results = rrf_fusion(all_ranked_lists)[:10]

    if not fused_results:
        return "Não encontrei trechos relevantes após fusão.", []

    # 5. Reordena por página — preserva ordem pedagógica do manual
    fused_results.sort(key=lambda x: x[0])

    # 6. Fontes para o frontend
    sources = [
        f"[Pág. {page}]: {texto[:150].strip()}..."
        for page, texto in fused_results
    ]

    # 7. Contexto para o LLM
    context = "\n\n".join(
        f"[Página {page}]: {texto}"
        for page, texto in fused_results
    )

    # 8. Prompt pedagógico
    system_prompt = (
        "Você é um monitor experiente de {game_title}. Sua missão é explicar as regras "
        "de forma didática, fluida e amigável. Use EXCLUSIVAMENTE os trechos fornecidos.\n\n"
        "DIRETRIZES DE RESPOSTA:\n"
        "- Responda de forma direta. Comece enunciando a regra principal com naturalidade.\n"
        "- Se a regra envolver passos ou uma sequência, use lista numerada ou bullets.\n"
        "- Se a regra for uma permissão, enuncie como permissão. Se for proibição, como proibição.\n"
        "- Mencione exceções apenas se existirem nos trechos, integrando-as ao texto.\n"
        "- Se a informação for insuficiente, informe honestamente que o manual não detalha esse ponto.\n"
        "- NUNCA utilize conhecimento externo ou invente regras que não estejam nos trechos."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "TRECHOS DO MANUAL (em ordem de página):\n{context}"),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question,
        "game_title": game_title
    })

    # 9. Adiciona fontes ao final da resposta
    paginas = []
    for page, _ in fused_results:
        if page not in paginas:
            paginas.append(page)
    paginas_str = ", ".join(f"Página {p}" for p in paginas)
    resposta_final = f"{response.content}\n\nFONTES: {paginas_str}"

    return resposta_final, sources