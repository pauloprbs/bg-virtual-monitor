import os
import streamlit as st
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Monitor de Board Games", page_icon="🎲")

def get_games():
    try:
        return requests.get(f"{BACKEND_URL}/games").json()
    except:
        return []

st.title("🎲 Monitor de Board Games")

with st.sidebar:
    games = get_games()
    game_choice = st.selectbox("Jogo:", games) if games else None
    if not games: st.error("Nenhum jogo no banco!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "sources" in m:
            with st.expander("Fontes"):
                for s in m["sources"]: st.info(s)

if prompt := st.chat_input("Dúvida?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        res = requests.post(f"{BACKEND_URL}/ask", json={"game_title": game_choice, "question": prompt})
        if res.status_code == 200:
            data = res.json()
            st.markdown(data["answer"])
            
            # MOSTRA OS TRECHOS RECUPERADOS (Exigência 7.2)
            sources = data.get("sources", [])
            with st.expander("🔍 Ver trechos recuperados do manual"):
                for s in sources: st.info(s)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": data["answer"], 
                "sources": sources
            })