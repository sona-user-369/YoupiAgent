"""RAG sur les documents de l'événement (dossier settings_yp.DOCUMENTS_DIR).

Pipeline en deux étages :
1. Récupération : les documents (.md / .txt / .pdf) sont découpés en passages,
   vectorisés avec le modèle d'embedding RodiumAI (API compatible OpenAI) et
   stockés dans Qdrant (mode local sur disque, comme pour Mem0). Une recherche
   vectorielle ramène RETRIEVE_K candidats.
2. Reranking : un cross-encoder multilingue (fastembed, exécuté en local)
   relit chaque paire (question, passage) et ne garde que les TOP_K meilleurs.

L'index se resynchronise automatiquement quand un document est ajouté,
modifié ou supprimé : il suffit de déposer un fichier, sans redémarrer l'agent.
"""

import hashlib
import threading
import uuid
from typing import Dict, List

from openai import OpenAI
from qdrant_client import QdrantClient, models

import settings_yp

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
CHUNK_SIZE = 900  # caractères
CHUNK_OVERLAP = 150
EMBED_BATCH = 64
RETRIEVE_K = 12  # candidats ramenés par la recherche vectorielle
TOP_K = 4  # passages gardés après reranking
COLLECTION = "youpi_event_documents"


def _read_document(path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk_text(text: str) -> List[str]:
    """Découpe en passages d'environ CHUNK_SIZE caractères, en respectant les paragraphes."""
    paragraphs = [p.strip() for p in text.replace("\r", "").split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) + 2 > CHUNK_SIZE:
            chunks.append(current)
            current = current[-CHUNK_OVERLAP:] + "\n\n" + p
        else:
            current = f"{current}\n\n{p}" if current else p
        while len(current) > CHUNK_SIZE * 2:  # paragraphe géant
            chunks.append(current[:CHUNK_SIZE])
            current = current[CHUNK_SIZE - CHUNK_OVERLAP :]
    if current:
        chunks.append(current)
    return chunks


class DocumentIndex:
    def __init__(self):
        self._lock = threading.Lock()
        self._client = OpenAI(
            api_key=settings_yp.RODIUMAI_APIKEY, base_url=settings_yp.RODIUMAI_BASE_URL
        )
        self._qdrant = QdrantClient(path=str(settings_yp.RAG_STORAGE_DIR))
        self._reranker = None  # chargé à la première recherche (téléchargement du modèle)
        self._signature = None

    def _get_reranker(self):
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(model_name=settings_yp.RAG_RERANKER_MODEL)
        return self._reranker

    def _files(self):
        settings_yp.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        return sorted(
            p
            for p in settings_yp.DOCUMENTS_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith(".")
        )

    def _current_signature(self, files) -> str:
        h = hashlib.sha256(settings_yp.RODIUMAI_EMBEDDING_MODEL.encode())
        for p in files:
            h.update(p.name.encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
        return h.hexdigest()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for i in range(0, len(texts), EMBED_BATCH):
            resp = self._client.embeddings.create(
                model=settings_yp.RODIUMAI_EMBEDDING_MODEL, input=texts[i : i + EMBED_BATCH]
            )
            vectors.extend(d.embedding for d in resp.data)
        return vectors

    def _stored_signature(self):
        if not self._qdrant.collection_exists(COLLECTION):
            return None
        points, _ = self._qdrant.scroll(COLLECTION, limit=1, with_payload=True)
        return points[0].payload.get("signature") if points else None

    def _refresh(self) -> None:
        files = self._files()
        signature = self._current_signature(files)
        if signature == self._signature:
            return
        if signature == self._stored_signature():
            self._signature = signature
            return

        # Les documents ont changé : on reconstruit la collection.
        if self._qdrant.collection_exists(COLLECTION):
            self._qdrant.delete_collection(COLLECTION)
        chunks = [
            {"source": p.name, "text": text}
            for p in files
            for text in _chunk_text(_read_document(p))
        ]
        if chunks:
            vectors = self._embed([f"{c['source']}\n{c['text']}" for c in chunks])
            self._qdrant.create_collection(
                COLLECTION,
                vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
            )
            self._qdrant.upsert(
                COLLECTION,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=v,
                        payload={**c, "signature": signature},
                    )
                    for c, v in zip(chunks, vectors)
                ],
            )
        self._signature = signature

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        with self._lock:
            self._refresh()
            if not self._qdrant.collection_exists(COLLECTION):
                return []
            hits = self._qdrant.query_points(
                COLLECTION, query=self._embed([query])[0], limit=RETRIEVE_K
            ).points
            if not hits:
                return []
            scores = list(self._get_reranker().rerank(query, [h.payload["text"] for h in hits]))
            ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:top_k]
            return [
                {"source": h.payload["source"], "score": round(float(s), 3), "text": h.payload["text"]}
                for h, s in ranked
            ]


_index = None
_index_lock = threading.Lock()


def get_document_index() -> DocumentIndex:
    global _index
    with _index_lock:
        if _index is None:
            _index = DocumentIndex()
        return _index
