import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="BG Virtual Monitor",
    page_icon="🎲",
    layout="centered"
)

# ── Funções auxiliares ────────────────────────────────────────────────────────
def get_games() -> list[str]:
    try:
        resp = requests.get(f"{BACKEND_URL}/games", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Não foi possível conectar à API. Verifique se o backend está rodando.")
        return []
    except Exception:
        st.error("Erro ao carregar a lista de jogos.")
        return []


def ask_question(game_title: str, question: str, mode: str) -> dict | None:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/ask",
            json={"game_title": game_title, "question": question, "mode": mode},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        st.error("A requisição demorou demais. O modelo pode estar carregando — tente novamente.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Erro da API ({e.response.status_code}): {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Erro inesperado: {e}")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎲 BG Virtual Monitor")
    st.caption("Monitor de regras com RAG — IFG Pós-IA PLN")
    st.divider()

    games = get_games()
    if not games:
        st.warning("Nenhum jogo disponível. Verifique se a ingestão foi executada.")
        game_choice = None
    else:
        game_choice = st.selectbox("📖 Jogo", games)

    st.divider()

    # Seletor de modo — Trilha A
    st.markdown("**Modo de Recuperação**")
    mode_labels = {
        "🔀 Híbrido (Denso + BM25)": "hibrido",
        "🔍 Denso (vetorial)": "denso",
        "🔤 Esparso (BM25)": "esparso",
    }
    mode_label = st.radio(
        "Modo",
        options=list(mode_labels.keys()),
        index=0,
        label_visibility="collapsed",
        help=(
            "**Híbrido:** combina busca vetorial e BM25 via RRF — melhor recall geral.\n\n"
            "**Denso:** busca por similaridade semântica — bom para perguntas abertas.\n\n"
            "**Esparso (BM25):** busca por palavra-chave — melhor para termos exatos como nomes de peças."
        )
    )
    mode = mode_labels[mode_label]

    st.divider()
    if st.button("🗑️ Limpar conversa"):
        st.session_state.messages = []
        st.rerun()

# ── Área principal ────────────────────────────────────────────────────────────
st.header("💬 Pergunte sobre as regras")

if not game_choice:
    st.info("Selecione um jogo na barra lateral para começar.")
    st.stop()

# Inicializa histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("🔍 Ver trechos recuperados do manual"):
                for s in msg["sources"]:
                    st.info(s)
            if msg.get("mode"):
                mode_display = {"hibrido": "Híbrido", "denso": "Denso", "esparso": "Esparso (BM25)"}
                st.caption(f"Modo usado: {mode_display.get(msg['mode'], msg['mode'])}")

# Input do usuário
if prompt := st.chat_input(f"Dúvida sobre {game_choice}?"):
    if not prompt.strip():
        st.stop()

    # Exibe mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chama a API e exibe resposta
    with st.chat_message("assistant"):
        with st.spinner("Consultando o manual..."):
            data = ask_question(game_choice, prompt, mode)

        if data:
            st.markdown(data["answer"])

            sources = data.get("sources", [])
            if sources:
                with st.expander("🔍 Ver trechos recuperados do manual"):
                    for s in sources:
                        st.info(s)

            mode_display = {"hibrido": "Híbrido", "denso": "Denso", "esparso": "Esparso (BM25)"}
            st.caption(f"Modo usado: {mode_display.get(mode, mode)}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "sources": sources,
                "mode": mode
            })