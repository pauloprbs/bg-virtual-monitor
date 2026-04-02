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
git clone git@github.com:pauloprbs/bg-virtual-monitor.git
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
└── Catan.pdf              →  "Catan"
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

### Como rodar

```bash
docker exec -it bg_virtual_monitor_app python scripts/evaluate.py
```

O script gera:
- Tabela de Recall@k (k=3, 5, 10) para os três modos de recuperação
- Avaliação RAGAS de faithfulness para cada pergunta do golden set
- Arquivo `ragas_results_<provider>.json` com resultados detalhados

### Recall@k — Trilha A (Brass Birmingham)

| Modo | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|
| Denso (baseline) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) |
| Esparso — BM25 | 0.75 (9/12) | 0.75 (9/12) | 0.75 (9/12) |
| Híbrido — RRF | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) |

O modo híbrido mantém o recall máximo do denso e adiciona robustez para termos exatos via BM25. O BM25 perde em perguntas puramente semânticas, mas ganha em termos técnicos específicos como nomes de peças e ações.

### Golden Set — Faithfulness por questão (Groq, Brass Birmingham)

Avaliação RAGAS com 20 perguntas agnósticas ao jogo. Faithfulness média: **0.7466**.

| ID | Pergunta | Faithfulness |
|---|---|---|
| Q01 | Como se consegue a vitória no jogo? | 0.91 ✅ |
| Q02 | Quantos jogadores podem participar? | 0.75 🟡 |
| Q03 | O que posso fazer/executar no meu turno? | 0.88 ✅ |
| Q04 | Como termina uma rodada? | 0.90 ✅ |
| Q05 | Como é determinada a ordem dos turnos? | 0.80 ✅ |
| Q06 | Quais são os componentes incluídos na caixa do jogo? | 1.00 ✅ |
| Q07 | Para que serve o tabuleiro principal? | 1.00 ✅ |
| Q08 | O que acontece quando um jogador não pode executar uma ação por falta de recursos? | 0.33 ⚠️ |
| Q09 | Como a pontuação final é calculada e quais elementos contribuem para ela? | 0.75 🟡 |
| Q10 | Quando posso passar meu turno e o que isso implica? | 0.50 ⚠️ |
| Q11 | Como deve ser feita a preparação (setup) inicial dos componentes no tabuleiro? | 0.67 🟡 |
| Q12 | O que acontece quando algum componente limitado do jogo é exaurido? | 0.75 🟡 |
| Q13 | Existe alguma condição que encerra o jogo antes do fim previsto? | 0.75 🟡 |
| Q14 | O manual oferece alguma dica ou orientação estratégica para os jogadores? | 0.95 ✅ |
| Q15 | Existe alguma expansão oficial para este jogo? | 0.00 ⚠️ |
| Q16 | Quanto tempo dura uma partida em média? | 1.00 ✅ |
| Q17 | O que acontece se um jogador cometer um erro em uma ação já executada? | 0.33 ⚠️ |
| Q18 | O que acontece se dois jogadores empatarem na ordem de turno? | 1.00 ✅ |
| Q19 | Posso executar a mesma ação mais de uma vez no mesmo turno? | 0.67 🟡 |
| Q20 | Quem é declarado vencedor em caso de empate na pontuação final? | 1.00 ✅ |

> **Nota sobre Q15 (0.00):** Esta pergunta é de recusa esperada — o manual não menciona expansões. O RAGAS penaliza a resposta porque o ground_truth gerado automaticamente divergiu da resposta de recusa do bot. Casos como Q08, Q10 e Q17 apresentam faithfulness baixo por falso negativo do avaliador: as respostas estão corretas, mas o ground_truth gerado pelo LLM puxou trechos diferentes do corpus.

### Comparativo de avaliadores (Brass Birmingham)

| Avaliador | Modelo | Faithfulness média |
|---|---|---|
| Groq | Llama 3.1 8B Instant | **0.7466** |
| Ollama (local) | Qwen 2.5 3B | 0.4092 |

A diferença entre avaliadores reflete a capacidade do modelo de verificar grounding — modelos menores tendem a ser mais permissivos ou inconsistentes como juízes RAGAS.

### Recall@k — Trilha A (Catan)

| Modo | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|
| Denso (baseline) | 0.75 (9/12) | 0.75 (9/12) | 0.92 (11/12) |
| Esparso — BM25 | 0.75 (9/12) | 0.83 (10/12) | 0.83 (10/12) |
| Híbrido — RRF | 0.67 (8/12) | 0.75 (9/12) | 0.83 (10/12) |

No manual do Catan, o modo Denso atingiu a maior performance isolada no $k=10$. Isso demonstra que a semântica das regras de Catan é bem capturada pelo modelo de embeddings bge-m3, superando a correspondência exata do BM25 em contextos de maior profundidade. O modo híbrido, embora robusto, apresentou uma leve queda no rankeamento inicial ($k=3$), sugerindo que a fusão RRF pode ser refinada para este manual específico.

### Golden Set — Faithfulness por questão (Groq, Catan)

Avaliação RAGAS com 20 perguntas agnósticas ao jogo. Faithfulness média: **0.5670.**.

| ID | Pergunta | Faithfulness |
|---|---|---|
| Q01 | Como se consegue a vitória no jogo? | 0.80 ✅ |
| Q02 | Quantos jogadores podem participar? | 0.60 🟡 |
| Q03 | O que posso fazer/executar no meu turno? | 0.50 ⚠️ |
| Q04 | Como termina uma rodada? | 0.40 ⚠️ |
| Q05 | Como é determinada a ordem dos turnos? | 1.00 ✅ |
| Q06 | Quais são os componentes incluídos na caixa do jogo? | 0.80 ✅ |
| Q07 | Para que serve o tabuleiro principal? | 0.50 ⚠️ |
| Q08 | O que acontece quando um jogador não pode executar uma ação por falta de recursos? | 1.00 ✅ |
| Q09 | Como a pontuação final é calculada e quais elementos contribuem para ela? | 0.25 ⚠️ |
| Q10 | Quando posso passar meu turno e o que isso implica? | 0.70 🟡 |
| Q11 | Como deve ser feita a preparação (setup) inicial dos componentes no tabuleiro? | 0.33 ⚠️ |
| Q12 | O que acontece quando algum componente limitado do jogo é exaurido? | 1.00 ✅ |
| Q13 | Existe alguma condição que encerra o jogo antes do fim previsto? | 0.20 ⚠️ |
| Q14 | O manual oferece alguma dica ou orientação estratégica para os jogadores? | 0.40 ⚠️ |
| Q15 | Existe alguma expansão oficial para este jogo? | 0.00 ⚠️ |
| Q16 | Quanto tempo dura uma partida em média? | 0.50 ⚠️ |
| Q17 | O que acontece se um jogador cometer um erro em uma ação já executada? | 0.40 ⚠️ |
| Q18 | O que acontece se dois jogadores empatarem na ordem de turno? | 1.00 ✅ |
| Q19 | Posso executar a mesma ação mais de uma vez no mesmo turno? | 0.06 ⚠️ |
| Q20 | Quem é declarado vencedor em caso de empate na pontuação final? | 1.00 ✅ |

Nota sobre Faithfulness no Catan: A média de 0.5670 no Catan foi inferior à do Brass (0.7466) devido à natureza mais concisa do manual original, o que gerou fragmentos (chunks) com menor densidade de detalhes técnicos para o avaliador. Q15 e Q19 apresentam scores críticos devido à ausência explícita dessas informações no manual; o bot corretamente recusa ou simplifica, mas o juiz RAGAS penaliza a falta de evidências textuais diretas no ground truth.

### Comparativo de avaliadores (Catan)

| Avaliador | Modelo | Faithfulness média |
|---|---|---|
| Groq | Llama 3.1 8B Instant | **0.5670** |
| Ollama (local) | Qwen 2.5 3B | 0.4000 |

A análise comparativa reforça que modelos maiores em nuvem (Groq) conseguem realizar uma verificação de grounding muito mais rigorosa. O Qwen 2.5 3B local apresenta uma tendência a ser excessivamente crítico ou inconsistente na atribuição de notas de fidelidade, resultando em um score de 0.40. Para fins de entrega final, os dados do Groq são considerados a base de verdade do projeto.

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

Alunos: Paulo Renato Baliza Silva, Carlos Augusto da Silva Cabral e Eduardo Martins de Castro Souza
Projeto desenvolvido para a disciplina de Processamento de Linguagem Natural  
Pós-Graduação em Inteligência Artificial Aplicada — Instituto Federal de Goiás (IFG)

---

## ⚖️ Licença

Este projeto está sob a licença **MIT**.
