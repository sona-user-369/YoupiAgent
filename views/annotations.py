"""Page Streamlit : réponses annotées (👍/👎) et export CSV."""

import pandas as pd
import streamlit as st

import database

st.title("🏷️ Annotations & export CSV")
st.caption(
    "Les 👍 / 👎 placés sous les réponses de l'agent sont enregistrés ici : "
    "👍 = correct, 👎 = incorrect. Exportez la conversation avec ses labels."
)

scope = st.radio("Portée", ["Conversation en cours", "Toutes les conversations"], horizontal=True)
turns = database.get_turns(
    st.session_state.session_id if scope == "Conversation en cours" else None
)

if not turns:
    st.info("Aucun échange à exporter.")
    st.stop()

LABELS = {"correct": "correct", "incorrect": "incorrect", None: "non labellisé"}
df = pd.DataFrame(
    {
        "session_id": [t["session_id"] for t in turns],
        "user_id": [t["user_id"] for t in turns],
        "timestamp": [t["timestamp"] for t in turns],
        "question": [t["user_text"] for t in turns],
        "reponse_agent": [t["assistant_text"] for t in turns],
        "label": [LABELS[t["label"]] for t in turns],
        "tokens_rag": [t["rag_tokens"] for t in turns],
        "appels_outils": [sum(t["tool_counts"].values()) for t in turns],
    }
)

counts = df["label"].value_counts()
c1, c2, c3 = st.columns(3)
c1.metric("👍 Correct", int(counts.get("correct", 0)))
c2.metric("👎 Incorrect", int(counts.get("incorrect", 0)))
c3.metric("Non labellisé", int(counts.get("non labellisé", 0)))

only_labeled = st.checkbox("Exporter uniquement les réponses labellisées")
export_df = df[df["label"] != "non labellisé"] if only_labeled else df

st.dataframe(export_df, width="stretch", hide_index=True)

st.download_button(
    "⬇️ Télécharger le CSV",
    # utf-8-sig : Excel ouvre correctement les accents.
    data=export_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="conversation_labels.csv",
    mime="text/csv",
    disabled=export_df.empty,
)
