"""Interface de discussion Streamlit pour YoupiAgent.

Lancer avec : streamlit run app_streamlit.py
"""

import asyncio
import uuid

import streamlit as st

import database
from memory import DEFAULT_MEMORY_TYPE, MEMORY_TYPES, get_store_for

st.set_page_config(page_title="YoupiAgent - Support événement", page_icon="🎫")

# Les pages Stats/Annotations lisent la base avant que l'agent ne soit créé.
database.init_db()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []
if "memory_type" not in st.session_state:
    st.session_state.memory_type = DEFAULT_MEMORY_TYPE
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "event_loop" not in st.session_state:
    # Le client RodiumAI est async-only et garde ses connexions HTTP ouvertes
    # sur la boucle asyncio qui les a créées. Streamlit ré-exécute ce script
    # à chaque interaction : `asyncio.run()` créerait et fermerait une
    # nouvelle boucle à chaque fois, cassant ces connexions ("Event loop is
    # closed"). On garde donc une seule boucle vivante pour toute la session.
    st.session_state.event_loop = asyncio.new_event_loop()

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
    # Le type de mémoire ne peut être choisi qu'avant le premier message d'une
    # conversation : changer en cours de route fausserait la comparaison.
    conversation_started = bool(st.session_state.history)
    st.session_state.memory_type = st.radio(
        "Type de mémoire",
        options=list(MEMORY_TYPES),
        format_func=MEMORY_TYPES.get,
        index=list(MEMORY_TYPES).index(st.session_state.memory_type),
        disabled=conversation_started,
        help="Modifiable uniquement avant de démarrer une nouvelle conversation.",
    )
    if conversation_started:
        st.caption("🔒 Verrouillé pour cette conversation. Cliquez sur « Nouvelle conversation » pour changer.")

    if st.button("Nouvelle conversation"):
        st.session_state.history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    if st.button(
        "🗑️ Flush memory",
        help="Supprime toute la mémoire de cet email : faits Mem0, messages en cache et logs de toutes les sessions.",
    ):
        flush_user_id = st.session_state.user_id.strip().lower()
        if not flush_user_id:
            st.warning("Renseignez d'abord votre email.")
        else:
            try:
                for memory_type in MEMORY_TYPES:
                    get_store_for(memory_type).flush_user(flush_user_id)
            except Exception as e:
                st.error(f"Échec de la suppression de la mémoire : {e}")
            else:
                st.session_state.history = []
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.flush_notice = f"Mémoire de {flush_user_id} supprimée."
                st.rerun()

    if "flush_notice" in st.session_state:
        st.success(st.session_state.pop("flush_notice"))

pages = [
    st.Page("views/chat.py", title="Conversation", icon="💬", default=True),
    st.Page("views/logs.py", title="Logs mémoire & tokens", icon="🔍"),
    st.Page("views/analytics.py", title="Stats RAG & outils", icon="📊"),
    st.Page("views/errors.py", title="Erreurs", icon="🚨"),
    st.Page("views/annotations.py", title="Annotations & export CSV", icon="🏷️"),
]
st.navigation(pages).run()
