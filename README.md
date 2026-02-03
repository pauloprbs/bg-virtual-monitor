# 🚀 BG Virtual Monitor
**O "Mestre de Regras" Digital para Jogadores de Tabuleiro Modernos.**

O **BG Virtual Monitor** é um ecossistema de inteligência projetado para atuar como o monitor de uma luderia: ele explica regras, resolve conflitos de interpretação de manuais e analisa o mercado de jogos. O projeto utiliza uma arquitetura de agentes focada em performance, custo zero de infraestrutura e persistência de dados.

---

## 🛠️ Stack Tecnológica & Objetivos de Estudo

### 1. Orquestração e Inteligência (O Cérebro)
* **LangGraph (Stateful Agents):** Estudo de agentes que mantêm o estado da conversa e decidem entre fluxos de explicação de regras ou consulta de mercado.
* **Groq Cloud (LLM):** Uso do modelo **Llama 3** para raciocínio lógico de alta velocidade com custo zero de API.

### 2. NLP & Retrieval (O Conhecimento)
* **BGE-M3 via Ollama (Embeddings):** Implementação local do modelo BGE-M3 (Hugging Face) para garantir privacidade e soberania de dados.
* **Semantic Chunking:** Técnica avançada de segmentação de documentos baseada em variação semântica, evitando cortes arbitrários em parágrafos de regras.
* **PostgreSQL + pgvector:** Persistência de dados vetoriais e relacionais, permitindo buscas por similaridade de cosseno via SQL.

### 3. Integração & Interface
* **MCP (Model Context Protocol):** Desenvolvimento de ferramentas para conexão em tempo real com as APIs do **BGG** (dados globais) e **Ludopedia** (preços no Brasil).
* **FastAPI & Pydantic:** Backend robusto com documentação técnica detalhada via Swagger.

---

## 🏗️ Arquitetura de Persistência

Diferente de sistemas RAG puramente em memória, este projeto foca em eficiência industrial:

1.  **Camada de Documentos:** Armazena metadados e arquivos originais no PostgreSQL.
2.  **Camada Vetorial:** Persiste os embeddings gerados. Uma vez que o manual de um jogo é processado, ele fica disponível permanentemente, economizando recursos de computação em consultas futuras.

---

## 📅 Roadmap de Desenvolvimento

- [ ] **Fase 1:** Setup da infraestrutura via Docker (Postgres/pgvector + Ollama).
- [ ] **Fase 2:** Pipeline de ingestão com Semantic Chunking e persistência vetorial.
- [ ] **Fase 3:** Desenvolvimento do grafo de agentes com LangGraph.
- [ ] **Fase 4:** Implementação do servidor MCP para busca de preços e mercado.

---

## ⚖️ Licença

Este projeto está sob a licença **MIT**. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.