"""Page Streamlit : logs de la mémoire Mem0 et des tokens envoyés au LLM."""

import streamlit as st

from memory import MEMORY_TYPES, get_store_for

st.title("🔍 Logs mémoire & tokens")

st.caption(
    "Ce que l'agent envoie réellement au LLM à chaque tour : le dernier "
    "échange de la session (fenêtre courte) + les faits Mem0 trouvés par "
    "recherche sémantique — jamais l'historique complet. Sert à vérifier "
    "que le nombre de tokens envoyés reste stable, même quand la "
    "conversation s'allonge."
)

st.caption(f"Type de mémoire de cette conversation : **{MEMORY_TYPES[st.session_state.memory_type]}**")
logs = get_store_for(st.session_state.memory_type).get_logs(st.session_state.session_id)

if not logs:
    st.info("Aucun échange pour l'instant dans cette session.")
else:
    token_totals = [entry["tokens"]["sent_to_llm"] for entry in logs]
    latest = logs[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Tokens envoyés (dernier tour)", token_totals[-1])
    col2.metric("Faits Mem0 trouvés (dernier tour)", len(latest["mem0_search"]["results"]))
    col3.metric("Tours dans cette session", len(logs))

    st.subheader("Tokens envoyés au LLM, tour après tour")
    st.line_chart({"tokens envoyés": token_totals})
    st.caption(
        "Cette courbe doit rester globalement plate au fil de la conversation : "
        "c'est la preuve que l'historique brut ne grossit plus le contexte."
    )

    st.subheader("Détail par tour")
    for i, entry in enumerate(reversed(logs)):
        turn_no = len(logs) - i
        preview = entry["user_text"][:70] + ("…" if len(entry["user_text"]) > 70 else "")
        with st.expander(f"Tour {turn_no} — « {preview} »", expanded=(i == 0)):
            st.markdown(f"**Message utilisateur :** {entry['user_text']}")

            is_classic = entry.get("memory_type") == "classic"
            if is_classic:
                st.caption("Mémoire classique : les 30 derniers messages sont injectés dans le system prompt (pas de recherche Mem0).")
            else:
                st.markdown("**Recherche Mem0** (`memory.search`, sémantique, filtrée par `user_id`)")
                st.code(f'query = "{entry["mem0_search"]["query"]}"', language="text")
            results = entry["mem0_search"]["results"]
            if is_classic:
                pass
            elif results:
                for r in results:
                    score = r.get("score")
                    score_txt = f"  _(score={score:.3f})_" if isinstance(score, (int, float)) else ""
                    st.markdown(f"- {r['memory']}{score_txt}")
            else:
                st.caption("Aucun fait pertinent trouvé pour ce participant.")

            events = entry.get("mem0_events") or []
            if events:
                st.markdown("**Faits extraits/mis à jour par Mem0 après ce tour**")
                for e in events:
                    st.markdown(f"- `{e['event']}` — {e['memory']}")

            short_term = entry.get("short_term_window") or []
            if is_classic:
                short_term = []
            st.markdown("**Fenêtre court-terme utilisée pour ce tour** (dernier échange, avant ce message)")
            if short_term:
                for m in short_term:
                    role = "utilisateur" if m["role"] == "user" else "assistant"
                    st.markdown(f"- _{role}_ : {m['content']}")
            elif is_classic:
                st.caption("Non utilisée : l'historique est dans la section mémoire du system prompt.")
            else:
                st.caption("Aucun échange précédent dans cette session (premier tour).")

            st.markdown("**Tokens envoyés au LLM pour ce tour**")
            tokens = entry["tokens"]
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("System prompt", tokens["system_prompt"])
            t2.metric("Dernier échange", tokens["short_term"])
            t3.metric("Message courant", tokens["current_message"])
            t4.metric("Total envoyé", tokens["sent_to_llm"])
