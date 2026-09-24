"""Page Streamlit : conversation avec l'agent."""

import asyncio

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from run import chat

st.title("🎫 YoupiAgent")
st.caption("Assistant de support client pour votre événement — inscriptions, programme et infos pratiques.")

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
                asyncio.set_event_loop(st.session_state.event_loop)
                reply = st.session_state.event_loop.run_until_complete(
                    chat(
                        user_input,
                        st.session_state.user_id.strip().lower(),
                        st.session_state.session_id,
                    )
                )
            st.markdown(reply)

        # L'historique ici ne sert qu'à l'affichage : on ne le repasse pas à
        # l'agent (voir run.chat / MemoryStore pour le contexte réellement
        # utilisé : dernier échange + faits Mem0).
        st.session_state.history.append(HumanMessage(content=user_input))
        st.session_state.history.append(AIMessage(content=reply))
