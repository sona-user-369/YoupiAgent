"""Outils métier pour l'agent de support d'événement.

Ces outils ne dépendent d'aucune API externe : les inscriptions sont gérées
dans une petite base SQLite locale (voir settings_yp.EVENT_DB_FILE) et le
programme de l'événement est statique, ce qui suffit pour la démonstration
et permet de tester l'agent hors-ligne.
"""

import re

from langchain_core.tools import tool

import database

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

EVENT_FAQ = {
    "lieu": "L'événement se tient au Centre de Conférences Horizon, 12 rue des Lumières.",
    "date": "L'événement a lieu le 15 octobre 2026, de 9h à 20h.",
    "parking": "Un parking gratuit est disponible sur place, entrée par la rue des Lumières.",
    "restauration": "Le déjeuner et les pauses café sont inclus. Merci de signaler vos restrictions alimentaires lors de l'inscription.",
    "badge": "Les badges sont à retirer à l'accueil dès 8h avec une pièce d'identité.",
}


@tool
def register_participant(name: str, email: str, sessions: list[str] | None = None) -> dict:
    """Inscrit un participant à l'événement, et éventuellement à des sessions précises.

    Args:
        name: Nom complet du participant.
        email: Adresse email du participant, utilisée comme identifiant unique.
        sessions: Liste d'identifiants de sessions (voir list_event_sessions) auxquelles
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
            "Utilisez list_event_sessions pour voir les sessions disponibles.",
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
def list_event_sessions() -> dict:
    """Liste toutes les sessions du programme de l'événement avec horaires et salles."""
    return {"status": "ok", "sessions": EVENT_SESSIONS}


@tool
def check_session_availability(session_id: str) -> dict:
    """Vérifie le nombre de places restantes pour une session donnée.

    Args:
        session_id: Identifiant de la session (voir list_event_sessions).
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
def get_event_faq(topic: str) -> dict:
    """Répond aux questions fréquentes sur l'événement (lieu, date, parking, restauration, badge).

    Args:
        topic: Le sujet de la question, par exemple 'lieu', 'date', 'parking', 'restauration' ou 'badge'.
    """
    key = topic.strip().lower()
    if key in EVENT_FAQ:
        return {"status": "ok", "answer": EVENT_FAQ[key]}
    return {
        "status": "not_found",
        "message": f"Pas d'information sur '{topic}'.",
        "available_topics": list(EVENT_FAQ.keys()),
    }


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
    list_event_sessions,
    check_session_availability,
    get_event_faq,
]
