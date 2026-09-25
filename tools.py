"""Outils métier pour l'agent de support d'événement.

Ces outils ne dépendent d'aucune API externe : les inscriptions sont gérées
dans une petite base SQLite locale (voir settings_yp.EVENT_DB_FILE) et le
programme de l'événement est statique, ce qui suffit pour la démonstration
et permet de tester l'agent hors-ligne.
"""

import re

from langchain_core.tools import tool

import database
from rag import get_document_index

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _invalid_email(email: str) -> dict:
    return {"status": "error", "message": f"L'adresse email '{email}' semble invalide."}


def _not_registered(email: str) -> dict:
    return {"status": "not_found", "message": f"Aucune inscription trouvée pour {email}."}


# Programme fixe de l'événement (données métier statiques).
EVENT_SESSIONS = {
    "keynote-ouverture": {
        "title": "Keynote d'ouverture",
        "start": "09:00",
        "room": "Amphithéâtre A",
        "capacity": 300,
    },
    "atelier-ia": {
        "title": "Atelier : construire un agent IA",
        "start": "11:00",
        "room": "Salle 2",
        "capacity": 40,
    },
    "table-ronde-securite": {
        "title": "Table ronde : sécurité et IA",
        "start": "14:00",
        "room": "Salle 3",
        "capacity": 80,
    },
    "networking": {
        "title": "Cocktail networking",
        "start": "18:00",
        "room": "Terrasse",
        "capacity": 500,
    },
}

@tool
def register_participant(name: str, email: str, sessions: list[str] | None = None) -> dict:
    """Inscrit un participant à l'événement, et éventuellement à des sessions précises.

    Args:
        name: Nom complet du participant.
        email: Adresse email du participant, utilisée comme identifiant unique.
        sessions: Liste d'identifiants de sessions (parmi : keynote-ouverture, atelier-ia, table-ronde-securite, networking) auxquelles
            inscrire le participant. Optionnel.
    """
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        return _invalid_email(email)

    sessions = sessions or []
    unknown = [s for s in sessions if s not in EVENT_SESSIONS]
    if unknown:
        return {
            "status": "error",
            "message": f"Session(s) inconnue(s) : {', '.join(unknown)}. "
            f"Sessions valides : {', '.join(EVENT_SESSIONS)}.",
        }

    capacities = {session_id: s["capacity"] for session_id, s in EVENT_SESSIONS.items()}
    try:
        database.upsert_participant(email, name, sessions, capacities)
    except database.SessionFullError as e:
        return {
            "status": "error",
            "message": f"La session '{EVENT_SESSIONS[e.session_id]['title']}' est complète.",
        }

    return {
        "status": "ok",
        "message": f"{name} est inscrit·e avec succès ({email}).",
        "sessions": sessions,
    }


@tool
def check_registration(email: str) -> dict:
    """Vérifie si un participant est déjà inscrit à l'événement, à partir de son email.

    À utiliser avant d'inscrire quelqu'un (register_participant), pour éviter les
    doublons, ou pour répondre simplement à une question du type "suis-je déjà
    inscrit ?".

    Args:
        email: Adresse email du participant à vérifier.
    """
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        return _invalid_email(email)

    participant = database.get_participant(email)
    if participant is None:
        return {"status": "ok", "registered": False, "message": f"Aucune inscription trouvée pour {email}."}
    return {
        "status": "ok",
        "registered": True,
        "name": participant["name"],
        "registered_at": participant["registered_at"],
        "message": f"{participant['name']} ({email}) est déjà inscrit·e.",
    }


@tool
def get_registration(email: str) -> dict:
    """Récupère les informations d'inscription d'un participant à partir de son email."""
    email = email.strip().lower()
    participant = database.get_participant(email)
    if participant is None:
        return _not_registered(email)
    return {"status": "ok", "participant": participant}


@tool
def cancel_registration(email: str) -> dict:
    """Annule l'inscription d'un participant à partir de son email."""
    email = email.strip().lower()
    if not database.delete_participant(email):
        return _not_registered(email)
    return {"status": "ok", "message": f"Inscription de {email} annulée."}


@tool
def check_session_availability(session_id: str) -> dict:
    """Vérifie le nombre de places restantes pour une session donnée.

    Args:
        session_id: Identifiant de la session (parmi : keynote-ouverture, atelier-ia, table-ronde-securite, networking).
    """
    if session_id not in EVENT_SESSIONS:
        return {"status": "error", "message": f"Session inconnue : {session_id}."}
    taken = database.count_session_registrations(session_id)
    session = EVENT_SESSIONS[session_id]
    return {
        "status": "ok",
        "session": session["title"],
        "capacity": session["capacity"],
        "registered": taken,
        "remaining": max(session["capacity"] - taken, 0),
    }


@tool
def search_event_documents(query: str) -> dict:
    """Recherche dans les documents officiels de l'événement (base de connaissances).

    Source de vérité pour TOUT ce qui concerne l'événement : son nom, sa description,
    les dates, le lieu, le programme, les intervenants, les tarifs, l'accès, le parking,
    la restauration, le badge, etc. À appeler avant de répondre à toute question sur
    l'événement, avec une requête précise et autonome.

    Args:
        query: Ce que l'on cherche, formulé en une phrase (ex: "nom et dates de l'événement").
    """
    results = get_document_index().search(query)
    if not results:
        return {
            "status": "not_found",
            "message": "Aucun document n'est disponible pour l'événement.",
        }
    return {"status": "ok", "passages": results}


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


BUSINESS_TOOLS = [
    register_participant,
    check_registration,
    get_registration,
    cancel_registration,
    check_session_availability,
    search_event_documents,
]
