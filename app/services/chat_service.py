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
    context_parts = []
    for i, r in enumerate(results):
        # Criamos uma referência clara que a IA pode citar
        content = r.content.strip().replace('\xa0', ' ')
        context_parts.append(f"--- TRECHO {i+1} (Página {r.page_number}) ---\n{content}")
    
    context = "\n\n".join(context_parts)

    # 4. Prompt Citando as fontes (Grounding)
    system_prompt = (
        "Você é um monitor especialista no jogo de tabuleiro {game_title}. " # Mantemos como variável
        "Sua tarefa é responder perguntas dos jogadores usando APENAS os trechos do manual fornecidos abaixo. "
        "\n\nREGRAS CRÍTICAS:\n"
        "1. CITAÇÃO OBRIGATÓRIA: Para cada regra mencionada, você DEVE indicar a página entre parênteses, ex: (Pág. 10).\n"
        "2. FIDELIDADE: Não invente regras. Se a informação não estiver nos trechos, responda: 'Lamentavelmente, não encontrei essa regra específica no manual de {game_title}.'\n"
        "3. LIMPEZA: Ignore linhas repetidas ou erros de formatação nos trechos.\n"
        "4. ESTILO: Seja direto, mas educado."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "TRECHOS DO MANUAL RECUPERADOS:\n{context}"),
        ("human", "{question}"),
    ])

    # 5. Chain e Invocação
    chain = prompt | llm
    
    response = chain.invoke({
        "context": context, 
        "question": question, 
        "game_title": game_title
    })
    
    return response.content