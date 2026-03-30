import os
import json
from sqlalchemy.orm import Session
from sqlalchemy import text
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from app.utils.text_processor import super_clean

# ── Embeddings ────────────────────────────────────────────────────────────────
embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://ollama:11434")

# ── Seleção de LLM via variável de ambiente ───────────────────────────────────
# Para usar o Groq:  LLM_PROVIDER=groq  (padrão)
# Para usar local:   LLM_PROVIDER=ollama  e  LLM_MODEL=qwen2.5:3b
#
# No .env ou docker-compose:
#   LLM_PROVIDER=groq
#   LLM_MODEL=llama-3.1-8b-instant   (ignorado quando provider=ollama usa LLM_MODEL)
#   LLM_PROVIDER=ollama
#   LLM_MODEL=qwen2.5:3b

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

if LLM_PROVIDER == "ollama":
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
    )
else:
    llm = ChatGroq(
        temperature=0,
        model_name=LLM_MODEL,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

# ── Constante do RRF ──────────────────────────────────────────────────────────
RRF_K = 60


def expand_query(question: str, game_title: str) -> list[str]:
    """Gera 1 variação da pergunta para ampliar o recall do retriever."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Você é um assistente que ajuda a melhorar buscas em manuais do jogo {game_title}. "
         "Dado uma pergunta, gere exatamente 1 variação usando vocabulário alternativo "
         "que possa aparecer no manual. "
         "Responda APENAS com um JSON array de string. Exemplo: [\"variacao 1\"]"),
        ("human", "{question}")
    ])
    chain = prompt | llm
    response = chain.invoke({"question": question, "game_title": game_title})
    try:
        variations = json.loads(response.content)
        if isinstance(variations, list):
            return [str(v) for v in variations[:1]]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def search_chunks(query: str, game_title: str, db: Session, limit: int = 6) -> list[tuple]:
    """Busca vetorial — retorna lista de (content, page_number)."""
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


def rrf_fusion(ranked_lists: list[list[tuple]], k: int = RRF_K) -> list[tuple]:
    """
    Reciprocal Rank Fusion (Cormack et al., 2009).
    Pontua cada chunk pela posição em cada lista e funde por score acumulado.
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


def get_answer(question: str, game_title: str, db: Session):
    # 1. Busca original
    original_results = search_chunks(question, game_title, db)

    if not original_results:
        return "Lamentavelmente, não encontrei informações sobre este jogo no manual.", []

    # 2. Query expansion (1 variação) + busca adicional
    variations = expand_query(question, game_title)
    all_ranked_lists = [original_results]
    for variation in variations:
        variation_results = search_chunks(variation, game_title, db)
        if variation_results:
            all_ranked_lists.append(variation_results)

    # 3. RRF — funde e pega top-6
    fused_results = rrf_fusion(all_ranked_lists)[:6]

    if not fused_results:
        return "Não encontrei trechos relevantes após fusão.", []

    # 4. Reordena por página — preserva ordem pedagógica do manual
    fused_results.sort(key=lambda x: x[0])

    # 5. Fontes para o frontend
    sources = [
        f"[Pág. {page}]: {texto[:150].strip()}..."
        for page, texto in fused_results
    ]

    # 6. Contexto para o LLM
    context = "\n\n".join(
        f"[Página {page}]: {texto}"
        for page, texto in fused_results
    )

    # 7. Prompt pedagógico — corrigido para não inverter permissões (fix Q19)
    system_prompt = (
        "Você é um monitor experiente do jogo de tabuleiro {game_title}, "
        "com habilidade de explicar regras de forma clara e progressiva para novos jogadores. "
        "Responda usando EXCLUSIVAMENTE os trechos do manual fornecidos abaixo.\n\n"
        "ESTRUTURA OBRIGATÓRIA DA RESPOSTA:\n"
        "1. REGRA GERAL: Explique o conceito principal exatamente como está nos trechos. "
        "Se a regra é uma permissão, enuncie como permissão. "
        "Se é uma proibição, enuncie como proibição. Nunca inverta o sentido.\n"
        "2. COMO FUNCIONA: Detalhe o funcionamento passo a passo, se aplicável.\n"
        "3. EXCEÇÕES: Mencione exceções ou casos especiais apenas se existirem nos trechos.\n\n"
        "REGRAS:\n"
        "- Siga sempre a ordem acima. Nunca misture exceções com a regra geral.\n"
        "- Se uma seção não se aplicar à pergunta, omita-a silenciosamente.\n"
        "- Não invente informações além do que está nos trechos.\n"
        "- Se a informação for insuficiente, diga apenas o que está disponível."
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

    # 8. Adiciona fontes ao final da resposta
    paginas = []
    for page, _ in fused_results:
        if page not in paginas:
            paginas.append(page)
    paginas_str = ", ".join(f"Página {p}" for p in paginas)
    resposta_final = f"{response.content}\n\nFONTES: {paginas_str}"

    return resposta_final, sources