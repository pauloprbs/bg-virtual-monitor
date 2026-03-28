import os
from sqlalchemy.orm import Session
from sqlalchemy import text
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from app.utils.text_processor import super_clean

embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://ollama:11434")

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def get_answer(question: str, game_title: str, db: Session):
    query_vector = embeddings.embed_query(question)

    sql = text("""
        SELECT c.content, c.page_number
        FROM game_chunks c
        JOIN games g ON c.game_id = g.id
        WHERE g.title ILIKE :title
        ORDER BY c.embedding <=> :vector
        LIMIT 6
    """)

    results = db.execute(sql, {"vector": str(query_vector), "title": f"%{game_title}%"}).fetchall()

    if not results:
        return "Lamentavelmente, não encontrei informações sobre este jogo no manual.", []

    # 1. Limpa e deduplica
    seen_texts = set()
    cleaned_results = []
    for r in results:
        texto_limpo = super_clean(r.content)
        # Fingerprint pelas primeiras 20 palavras — mais robusto que slice de chars
        fingerprint = " ".join(texto_limpo.split()[:20])
        if fingerprint in seen_texts:
            continue
        seen_texts.add(fingerprint)
        cleaned_results.append((r.page_number, texto_limpo))

    if not cleaned_results:
        return "Não encontrei trechos relevantes após limpeza.", []

    # 2. REORDENA POR PÁGINA — preserva a ordem pedagógica do manual
    cleaned_results.sort(key=lambda x: x[0])

    # 3. Fontes para o frontend (texto limpo, sem \n)
    sources = [
        f"[Pág. {page}]: {texto[:150].strip()}..."
        for page, texto in cleaned_results
    ]

    # 4. Contexto para o LLM na ordem correta do manual
    context = "\n\n".join(
        f"[Página {page}]: {texto}"
        for page, texto in cleaned_results
    )

    # 5. Prompt com estrutura pedagógica explícita
    system_prompt = (
        "Você é um monitor experiente do jogo de tabuleiro {game_title}, "
        "com habilidade de explicar regras de forma clara e progressiva para novos jogadores. "
        "Responda usando EXCLUSIVAMENTE os trechos do manual fornecidos abaixo.\n\n"
        "ESTRUTURA OBRIGATÓRIA DA RESPOSTA:\n"
        "1. REGRA GERAL: Explique o conceito principal de forma direta.\n"
        "2. COMO FUNCIONA: Detalhe o funcionamento passo a passo, se aplicável.\n"
        "3. EXCEÇÕES: Mencione exceções ou casos especiais apenas se existirem nos trechos.\n"
        "4. FONTE: Indique a(s) página(s) usada(s), ex: (Pág. 6).\n\n"
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

    return response.content, sources