"""Page Streamlit : tokens consommés par le RAG et appels d'outils, tour après tour."""

import pandas as pd
import streamlit as st

import database

st.title("📊 Stats RAG & outils")
st.caption(
    "Pour la conversation en cours : tokens des passages RAG renvoyés au LLM "
    "(résultats de `search_event_documents`) et nombre d'appels d'outils, "
    "tour après tour."
)

turns = database.get_turns(st.session_state.session_id)

if not turns:
    st.info("Aucun échange pour l'instant dans cette session.")
    st.stop()

df = pd.DataFrame(
    {
        "tour": range(1, len(turns) + 1),
        "tokens RAG": [t["rag_tokens"] for t in turns],
        "tokens envoyés au LLM": [t["tokens_sent"] for t in turns],
        "appels d'outils": [sum(t["tool_counts"].values()) for t in turns],
    }
).set_index("tour")
df["tokens RAG cumulés"] = df["tokens RAG"].cumsum()
df["appels d'outils cumulés"] = df["appels d'outils"].cumsum()

tool_df = (
    pd.DataFrame([t["tool_counts"] for t in turns], index=df.index).fillna(0).astype(int)
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tokens RAG (total)", int(df["tokens RAG"].sum()))
c2.metric("Tokens RAG (dernier tour)", int(df["tokens RAG"].iloc[-1]))
c3.metric("Appels d'outils (total)", int(df["appels d'outils"].sum()))
c4.metric("Appels RAG (total)", int(tool_df.get("search_event_documents", pd.Series([0])).sum()))

st.subheader("Tokens RAG par tour")
st.bar_chart(df[["tokens RAG"]])

st.subheader("Tokens RAG cumulés")
st.line_chart(df[["tokens RAG cumulés"]])

st.subheader("Appels d'outils par tour")
if tool_df.empty or tool_df.to_numpy().sum() == 0:
    st.caption("Aucun outil appelé pour l'instant.")
else:
    st.bar_chart(tool_df)

    st.subheader("Appels d'outils cumulés")
    st.line_chart(tool_df.cumsum())

    st.subheader("Total par outil")
    st.dataframe(
        tool_df.sum().rename("appels").sort_values(ascending=False).to_frame(),
        width="stretch",
    )
