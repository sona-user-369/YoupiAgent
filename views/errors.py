"""Page Streamlit : erreurs « sourdines » de l'agent (réponses vides, pannes avalées)."""

import pandas as pd
import streamlit as st

import database

st.title("🚨 Erreurs")
st.caption(
    "Tout ce qui a mal tourné sans que l'utilisateur le voie : réponse vide du LLM, "
    "boucle d'outils sans réponse finale, outil en échec, recherche ou écriture Mem0 "
    "qui échoue silencieusement, plantage de l'agent."
)

scope = st.radio("Portée", ["Conversation en cours", "Toutes les conversations"], horizontal=True)
session_filter = st.session_state.session_id if scope == "Conversation en cours" else None
errors = database.get_errors(session_filter)

if not errors:
    st.success("Aucune erreur enregistrée.")
    st.stop()

df = pd.DataFrame(errors)
n_err = int((df["level"] == "error").sum())
c1, c2 = st.columns(2)
c1.metric("🔴 Erreurs (l'utilisateur est impacté)", n_err)
c2.metric("🟠 Avertissements (dégradation silencieuse)", len(df) - n_err)

st.subheader("Par source")
st.bar_chart(df["source"].value_counts())

sources = sorted(df["source"].unique())
selected = st.multiselect("Filtrer par source", sources, default=sources)
shown = [e for e in errors if e["source"] in selected]

st.subheader("Détail (plus récentes en premier)")
for e in shown:
    icon = "🔴" if e["level"] == "error" else "🟠"
    with st.expander(f"{icon} {e['timestamp'][:19].replace('T', ' ')} — {e['source']} — {e['message'][:90]}"):
        st.markdown(f"**Message :** {e['message']}")
        st.caption(
            f"session `{e['session_id']}` · participant `{e['user_id']}` · tour `{e['turn_id']}`"
        )
        if e["details"]:
            st.code(e["details"], language="text")

if st.button("🗑️ Vider ces erreurs"):
    database.clear_errors(session_filter)
    st.rerun()
