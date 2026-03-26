import os
from sqlalchemy.orm import Session
from sqlalchemy import text
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Inicializa os componentes
embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://ollama:11434")

# Certifique-se de ter a GROQ_API_KEY no seu .env
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant", 
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def get_answer(question: str, game_title: str, db: Session):
    # 1. Gerar o embedding da pergunta
    query_vector = embeddings.embed_query(question)
    
    # 2. Busca vetorial filtrada por jogo
    # JOIN com a tabela 'games' para garantir que os trechos sejam do jogo certo
    sql = text("""
        SELECT c.content, c.page_number 
        FROM game_chunks c
        JOIN games g ON c.game_id = g.id
        WHERE g.title ILIKE :title
        ORDER BY c.embedding <=> :vector 
        LIMIT 4
    """)
    
    # Executa a busca passando o título do jogo
    results = db.execute(sql, {
        "vector": str(query_vector),
        "title": f"%{game_title}%" 
    }).fetchall()
    
    # Validação: Se o jogo não existir ou não tiver chunks
    if not results:
        return f"Desculpe, não encontrei o manual do jogo '{game_title}' no meu sistema."

    # 3. Montagem do contexto (Limpando o ruídos, como duplicações nos chunks)
    context_parts = [f"[Pág {r.page_number}]: {r.content.strip()}" for r in results]
    context = "\n\n".join(context_parts)

    # 4. Prompt e LLM (O Groq vai "limpar" as repetições de texto para você)
    system_prompt = (
        f"Você é um monitor especialista no jogo {game_title}. "
        "Use os trechos do manual fornecidos para responder. Ignore repetições de texto no contexto."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "CONTEXTO:\n{context}"),
        ("human", "{question}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    return response.content