# 🎲 BG Virtual Monitor
**Monitor virtual de regras para jogos de tabuleiro — powered by RAG.**

O **BG Virtual Monitor** é um chatbot baseado em **RAG (Retrieval-Augmented Generation)** que responde perguntas sobre regras de jogos de tabuleiro consultando os manuais oficiais indexados. Desenvolvido como projeto final da disciplina de Processamento de Linguagem Natural — Pós-Graduação em IA Aplicada (IFG).

---

## 🏗️ Arquitetura

```
Pergunta do usuário
        │
        ▼
  Query Expansion (LLM gera 2 variações)
        │
        ├── Busca Densa (BGE-M3 + pgvector)  ──┐
        ├── Busca Densa (variação 1)            ├── RRF Fusion
        ├── Busca Densa (variação 2)            │   (Reciprocal Rank Fusion)
        └── Busca Esparsa (BM25)             ──┘
                                                │
                                          Top-10 chunks
                                          (deduplicados)
                                                │
                                          Prompt + LLM
                                                │
                                        Resposta + Fontes
```

**Trilha A implementada:** Recuperação Híbrida (Sparse + Dense) com RRF, comparando os três modos (denso, esparso, híbrido) via Recall@k.

---

## 🛠️ Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| LLM (nuvem) | Groq — Llama 3.1 8B Instant |
| LLM (local) | Ollama — Qwen 2.5 3B |
| Embeddings | BGE-M3 via Ollama |
| Retriever denso | pgvector (similaridade de cosseno) |
| Retriever esparso | BM25 (rank-bm25) |
| Fusão | Reciprocal Rank Fusion (RRF, k=60) |
| Banco de dados | PostgreSQL 16 + pgvector |
| Backend | FastAPI + Pydantic |
| Frontend | Streamlit |
| Infraestrutura | Docker + Docker Compose |
| Avaliação | RAGAS + Recall@k manual |

---

## 📁 Estrutura do Projeto

```
bg-virtual-monitor/
├── app/
│   ├── database/
│   │   ├── database.py        # Conexão SQLAlchemy
│   │   └── models.py          # Modelos Game e GameChunk
│   ├── services/
│   │   └── chat_service.py    # Pipeline RAG (BM25 + denso + RRF + LLM)
│   ├── utils/
│   │   └── text_processor.py  # Limpeza de texto extraído de PDF
│   └── frontend.py            # Interface Streamlit
├── scripts/
│   ├── ingest.py              # Ingestão e chunking de manuais PDF
│   └── evaluate.py            # Avaliação: Recall@k (3 modos) + RAGAS
├── data/
│   └── manuals/               # PDFs dos manuais de jogos (colocar aqui)
├── main.py                    # FastAPI — endpoints /ask, /games
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env
```

---

## 🚀 Como Executar

### Pré-requisitos

- Docker e Docker Compose instalados
- Chave de API do Groq (gratuita em [console.groq.com](https://console.groq.com)) — opcional se usar modo local

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/bg-virtual-monitor.git
cd bg-virtual-monitor
```

### 2. Configure o `.env`

Copie o arquivo de exemplo e edite conforme necessário:

```bash
cp .env.example .env
```

Conteúdo mínimo do `.env`:

```env
# Banco de dados
POSTGRES_USER=admin
POSTGRES_PASSWORD=password
POSTGRES_DB=bgvirtualmonitordb
DATABASE_URL=postgresql://admin:password@db:5432/bgvirtualmonitordb

# LLM — escolha um dos modos abaixo:

# Modo nuvem (Groq):
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=sua_chave_aqui

# Modo local (Ollama — sem necessidade de API key):
# LLM_PROVIDER=ollama
# LLM_MODEL=qwen2.5:3b
```

### 3. Adicione os manuais

Coloque os arquivos PDF na pasta `data/manuals/`. O nome do arquivo vira o título do jogo:

```
data/manuals/
├── Brass_Birmingham.pdf   →  "Brass Birmingham"
├── Catan.pdf              →  "Catan"
└── Azul.pdf               →  "Azul"
```

### 4. Suba o projeto

```bash
docker compose up
```

Na **primeira execução**, o Docker irá:
1. Baixar os modelos BGE-M3 e Qwen 2.5 via Ollama (pode demorar ~10 min dependendo da conexão)
2. Inicializar o banco PostgreSQL com pgvector
3. Ingerir automaticamente todos os manuais da pasta `data/manuals/`
4. Subir a API (porta 8000) e o frontend Streamlit (porta 8501)

Nas execuções seguintes, os modelos e os dados já estão em cache — o sistema sobe em segundos.

### 5. Acesse

| Serviço | URL |
|---|---|
| Frontend (Streamlit) | http://localhost:8501 |
| API (FastAPI Swagger) | http://localhost:8000/docs |

---

## 📊 Avaliação

### Recall@k — Trilha A (3 modos)

Execute dentro do container:

```bash
docker exec -it bg_virtual_monitor_app python scripts/evaluate.py
```

O script gera automaticamente:
- Tabela de Recall@k (k=3, 5 e 10) para os modos **denso**, **esparso** e **híbrido**
- Avaliação RAGAS (faithfulness) com o LLM configurado no `.env`
- Arquivo `ragas_results_<provider>.json` com resultados por pergunta

### Resultados obtidos (Brass Birmingham)

| Modo | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|
| Denso | 1.00 | 1.00 | 1.00 |
| Esparso (BM25) | 0.75 | 0.75 | 0.75 |
| Híbrido (RRF) | 1.00 | 1.00 | 1.00 |

| Avaliador | Faithfulness média |
|---|---|
| Groq (Llama 3.1 8B) | 0.6892 |
| Ollama (Qwen 2.5 3B) | 0.4092 |

> O modo híbrido mantém o recall máximo do modo denso enquanto adiciona robustez para termos exatos via BM25. O modo esparso perde em perguntas semânticas mas ganha em termos técnicos específicos (ex: nomes de peças, ações). Análise completa no relatório.

---

## ⚙️ Parâmetros do Pipeline

| Parâmetro | Valor | Justificativa |
|---|---|---|
| chunk_size | 1000 chars | Preserva contexto suficiente sem ultrapassar janela do LLM |
| chunk_overlap | 100 chars | Evita perda de informação em bordas de chunk |
| k (retrieval) | 10 por lista | Cobre casos de perguntas multi-trecho |
| RRF_K | 60 | Valor padrão da literatura (Cormack et al., 2009) |
| top final | 10 chunks | Balanceia contexto rico com custo de tokens |
| Variações de query | 2 | Reduz consumo de tokens mantendo recall |

---

## 🔧 Comandos Úteis

```bash
# Reiniciar apenas a aplicação (após atualizar chat_service.py)
docker compose restart app

# Ingerir manualmente (se necessário)
docker exec -it bg_virtual_monitor_app python scripts/ingest.py

# Ver logs em tempo real
docker compose logs -f app

# Rodar avaliação com Groq
docker exec -it bg_virtual_monitor_app python scripts/evaluate.py

# Acessar o banco diretamente
docker exec -it postgres psql -U admin -d bgvirtualmonitordb
```

---

## 👥 Equipe

Projeto desenvolvido para a disciplina de Processamento de Linguagem Natural  
Pós-Graduação em Inteligência Artificial Aplicada — Instituto Federal de Goiás (IFG)

---

## ⚖️ Licença

Este projeto está sob a licença **MIT**.
