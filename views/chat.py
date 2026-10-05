"""Page Streamlit : conversation avec l'agent."""

import asyncio

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import database
from run import chat_turn

_LABEL_TO_FEEDBACK = {"correct": 1, "incorrect": 0}
_FEEDBACK_TO_LABEL = {1: "correct", 0: "incorrect"}


def _save_feedback(turn_id: str) -> None:
    value = st.session_state.get(f"feedback_{turn_id}")
    database.set_turn_label(turn_id, _FEEDBACK_TO_LABEL.get(value))


def _render_feedback(turn_id: str, labels: dict) -> None:
    """👍 / 👎 sous une réponse de l'agent, enregistré en base (correct / incorrect)."""
    key = f"feedback_{turn_id}"
    if key not in st.session_state and labels.get(turn_id) in _LABEL_TO_FEEDBACK:
        st.session_state[key] = _LABEL_TO_FEEDBACK[labels[turn_id]]
    st.feedback("thumbs", key=key, on_change=_save_feedback, args=(turn_id,))


st.title("🎫 YoupiAgent")
st.caption("Assistant de support client pour votre événement — inscriptions, programme et infos pratiques.")

labels = {t["turn_id"]: t["label"] for t in database.get_turns(st.session_state.session_id)}

for message in st.session_state.history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage) and message.content:
        with st.chat_message("assistant"):
            st.markdown(message.content)
            turn_id = message.additional_kwargs.get("turn_id")
            if turn_id:
                _render_feedback(turn_id, labels)

user_input = st.chat_input("Écrivez votre message...")

if user_input:
    if not st.session_state.user_id.strip():
        st.warning("Merci de renseigner votre email dans la barre latérale avant de discuter.")
    else:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            reply, turn_id = "", None
            with st.spinner("L'agent réfléchit..."):
                asyncio.set_event_loop(st.session_state.event_loop)
                try:
                    reply, turn_id = st.session_state.event_loop.run_until_complete(
                        chat_turn(
                            user_input,
                            st.session_state.user_id.strip().lower(),
                            st.session_state.session_id,
                            st.session_state.memory_type,
                        )
                    )
                except Exception as e:
                    st.error(f"L'agent a rencontré une erreur : {e}\n\nDétails dans l'onglet 🚨 Erreurs.")
            if reply:
                st.markdown(reply)
                _render_feedback(turn_id, {})
            elif turn_id:
                st.warning("L'agent n'a produit aucune réponse. Détails dans l'onglet 🚨 Erreurs.")

        # L'historique ici ne sert qu'à l'affichage : on ne le repasse pas à
        # l'agent (voir run.chat / MemoryStore pour le contexte réellement
        # utilisé : dernier échange + faits Mem0).
        st.session_state.history.append(HumanMessage(content=user_input))
        if reply:
            st.session_state.history.append(AIMessage(content=reply, additional_kwargs={"turn_id": turn_id}))
