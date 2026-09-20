"""Interface de discussion Streamlit pour YoupiAgent.

Lancer avec : streamlit run app_streamlit.py
"""

import asyncio
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from run import chat

st.set_page_config(page_title="YoupiAgent - Support événement", page_icon="🎫")

st.title("🎫 YoupiAgent")
st.caption("Assistant de support client pour votre événement — inscriptions, programme et infos pratiques.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

with st.sidebar:
    st.subheader("Votre identité")
    st.session_state.user_id = st.text_input(
        "Email (utilisé comme identifiant participant pour la mémoire)",
        value=st.session_state.user_id,
        placeholder="prenom.nom@example.com",
    )
    st.caption(
        "L'agent se souvient des informations que vous lui donnez (préférences, "
        "inscriptions, restrictions alimentaires...) d'une conversation à l'autre, "
        "tant que vous utilisez le même email."
    )
    if st.button("Nouvelle conversation"):
        st.session_state.history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

for message in st.session_state.history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage) and message.content:
        with st.chat_message("assistant"):
            st.markdown(message.content)

user_input = st.chat_input("Écrivez votre message...")

if user_input:
    if not st.session_state.user_id.strip():
        st.warning("Merci de renseigner votre email dans la barre latérale avant de discuter.")
    else:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("L'agent réfléchit..."):
                reply, updated_history = asyncio.run(
                    chat(
                        user_input,
                        st.session_state.history,
                        st.session_state.user_id.strip().lower(),
                        st.session_state.session_id,
                    )
                )
            st.markdown(reply)

        st.session_state.history = updated_history
