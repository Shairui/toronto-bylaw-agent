"""
rag.py — Retrieval-Augmented Generation knowledge base for Toronto bylaws.

Knowledge base sources (loaded at startup):
  excel/Waste Wizard Lookup Table.json   — 2,206 official waste disposal items,
                                            grouped by bin category into RAG documents.
  excel/Cleared Building Permits since 2017.json — 392k permit records; a sample of
                                            recent approved/issued permits is indexed.
  Inline regulatory bylaws               — ~25 key Toronto bylaw texts covering zoning,
                                            noise, parking, property standards, parks,
                                            trees, and business licensing (no external
                                            source file exists for these).

ChromaDB is wiped and rebuilt on every startup to avoid schema conflicts.
Falls back to keyword search when ChromaDB is empty or a query fails.
RAG results include cosine distance so callers can filter by relevance threshold.
"""
import json
import os
import csv
import re
import requests
from typing import List, Dict, Any
from pathlib import Path
import chromadb
from backend.config import (
    CHROMA_DB_PATH, KNOWLEDGE_BASE_PATH,
    EMBEDDING_API_KEY, EMBEDDING_API_BASE, EMBEDDING_MODEL,
)

_EXCEL_DIR = Path(__file__).parent.parent / "excel"
_WASTE_JSON = _EXCEL_DIR / "Waste Wizard Lookup Table.json"
_PERMITS_JSON = _EXCEL_DIR / "Cleared Building Permits since 2017.json"
_BYLAWS_JSON = _EXCEL_DIR / "regulatory_bylaws.json"


# Remote embedding function

class _RemoteEmbeddingFn:
    def __init__(self):
        self.url = f"{EMBEDDING_API_BASE.rstrip('/')}/embeddings"
        self.headers = {
            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
            "Content-Type": "application/json",
        }

    def __call__(self, input: List[str]) -> List[List[float]]:
        resp = requests.post(
            self.url, headers=self.headers,
            json={"model": EMBEDDING_MODEL, "input": input}, timeout=30,
        )
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]


def _probe_embedding_api() -> bool:
    if not EMBEDDING_API_KEY:
        return False
    try:
        url = f"{EMBEDDING_API_BASE.rstrip('/')}/embeddings"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}", "Content-Type": "application/json"},
            json={"model": EMBEDDING_MODEL, "input": ["test"]},
            timeout=5,
        )
        return resp.status_code < 500
    except Exception:
        return False


# RAG system

class TorontoBylawRAG:

    def __init__(self):
        collection_kwargs = {"name": "toronto_bylaws", "metadata": {"hnsw:space": "cosine"}}
        if _probe_embedding_api():
            collection_kwargs["embedding_function"] = _RemoteEmbeddingFn()
            print("[RAG] Using remote embedding API.")
        else:
            print("[RAG] Using local default embeddings.")

        self.client = self._create_client()
        self.collection = self.client.get_or_create_collection(**collection_kwargs)
        self.knowledge_base = self._load_knowledge_base()
        self.waste_items = self._load_waste_items()

    @staticmethod
    def _create_client() -> chromadb.PersistentClient:
        import shutil
        try:
            chromadb.api.client.SharedSystemClient._identifier_to_system.clear()
        except Exception:
            pass
        shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        return chromadb.PersistentClient(path=CHROMA_DB_PATH)

    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        if os.path.exists(KNOWLEDGE_BASE_PATH):
            with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._build_knowledge_base()

    # Knowledge base builder

    def _build_knowledge_base(self) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        docs.extend(self._load_waste_docs())
        docs.extend(self._load_permit_docs())
        docs.extend(self._regulatory_bylaws())
        print(f"[RAG] Built knowledge base: {len(docs)} documents")
        return docs

    # Waste Wizard (from excel file)

    @staticmethod
    def _load_waste_docs() -> List[Dict[str, Any]]:
        if not _WASTE_JSON.exists():
            print(f"[RAG] Waste Wizard file not found: {_WASTE_JSON}")
            return []

        with open(_WASTE_JSON, encoding="utf-8") as f:
            items = json.load(f)

        # Normalise category names
        _CAT_MAP = {
            "Oversized item": "Oversized Item",
            "Not Accepted ": "Not Accepted",
            "Not accepted": "Not Accepted",
            "Garbage": "Garbage Bin",
        }
        groups: Dict[str, List[str]] = {}
        for item in items:
            cat = _CAT_MAP.get(item["category"], item["category"])
            name = item["item"].split("/")[0].strip().title()
            instructions = " ".join(item.get("instructions", []))
            line = f"{name}: {instructions}" if instructions else name
            groups.setdefault(cat, []).append(line)

        _CAT_SOURCE = {
            "Blue Bin":      "toronto.ca/bluebin",
            "Green Bin":     "toronto.ca/greenbin",
            "Garbage Bin":   "toronto.ca/garbage",
            "HHW":           "toronto.ca/hhw",
            "Electronic Waste": "toronto.ca/electronics",
            "Oversized Item": "toronto.ca/bulkpickup",
            "Depot":         "toronto.ca/hhw",
            "Yard Waste":    "toronto.ca/yardwaste",
            "Metal":         "toronto.ca/bluebin",
            "Not Accepted":  "toronto.ca/garbage",
            "Christmas Tree": "toronto.ca/yardwaste",
        }

        docs = []
        for i, (cat, lines) in enumerate(groups.items()):
            docs.append({
                "id": f"waste_{i:03d}",
                "title": f"Waste Disposal — {cat}",
                "content": f"Items accepted in {cat}: " + "; ".join(lines),
                "source": f"City of Toronto Waste Wizard — {_CAT_SOURCE.get(cat, 'toronto.ca/garbage')}",
                "category": "Waste",
            })

        print(f"[RAG] Loaded {len(docs)} waste categories from Waste Wizard.")
        return docs

    @staticmethod
    def _load_waste_items() -> List[Dict[str, Any]]:
        if not _WASTE_JSON.exists():
            return []
        with open(_WASTE_JSON, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _normalise_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    @staticmethod
    def _keyword_tokens(text: str) -> List[str]:
        stopwords = {
            "where", "does", "do", "go", "goes", "put", "place", "which", "what",
            "bin", "bins", "waste", "garbage", "trash", "recycle", "recycling",
            "dispose", "disposal", "collection", "day", "old", "have", "has",
            "the", "this", "that", "into", "for", "with", "and", "toronto",
        }
        tokens = []
        for token in TorontoBylawRAG._normalise_text(text).split():
            if len(token) <= 2 or token in stopwords:
                continue
            tokens.append(token[:-1] if token.endswith("s") and len(token) > 4 else token)
        return tokens

    def lookup_waste_item(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find likely Waste Wizard item matches without relying on the LLM."""
        query_norm = self._normalise_text(query)
        query_tokens = self._keyword_tokens(query)
        if not query_tokens:
            return []

        scored = []
        for item in self.waste_items:
            item_name = item.get("item", "")
            item_norm = self._normalise_text(item_name)
            item_tokens = self._keyword_tokens(item_name)
            score = 0

            if item_norm and item_norm in query_norm:
                score += 20
            if query_norm and query_norm in item_norm:
                score += 12

            for token in query_tokens:
                if token in item_tokens:
                    score += 6
                elif token in item_norm:
                    score += 3

            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: (-x[0], len(x[1].get("item", ""))))
        return [item for _, item in scored[:top_k]]

    # Building permits (from excel file)

    @staticmethod
    def _load_permit_docs() -> List[Dict[str, Any]]:
        if not _PERMITS_JSON.exists():
            print(f"[RAG] Permits file not found: {_PERMITS_JSON}")
            return []

        with open(_PERMITS_JSON, encoding="utf-8") as f:
            records = json.load(f)

        # Keep only Approved/Issued-equivalent records with a street name
        keep_statuses = {"Approved", "Permit Issued", "Ready for Issuance", "Issuance Pending"}
        filtered = [
            r for r in records
            if r.get("STATUS") in keep_statuses and r.get("STREET_NAME")
        ]
        # Sort by application date descending, take top 150
        filtered.sort(key=lambda r: r.get("APPLICATION_DATE") or "", reverse=True)
        filtered = filtered[:150]

        docs = []
        for i, r in enumerate(filtered):
            street = f"{r.get('STREET_NUM', '')} {r.get('STREET_NAME', '')} {r.get('STREET_TYPE', '')}".strip()
            content = (
                f"Permit No. {r.get('PERMIT_NUM', 'N/A')} — Status: {r.get('STATUS', 'N/A')}. "
                f"Address: {street}, Toronto. "
                f"Type: {r.get('PERMIT_TYPE', 'N/A')}. "
                f"Work: {r.get('WORK') or r.get('DESCRIPTION') or 'N/A'}. "
                f"Application Date: {r.get('APPLICATION_DATE', 'N/A')}."
            )
            docs.append({
                "id": f"permit_{i:03d}",
                "title": f"Building Permit — {r.get('PERMIT_TYPE', 'N/A')}, {street}",
                "content": content,
                "source": "City of Toronto Open Data — Building Permits; toronto.ca/buildingpermits",
                "category": "Permit",
            })

        print(f"[RAG] Loaded {len(docs)} permit records.")
        return docs

    # Regulatory bylaw texts (loaded from excel/regulatory_bylaws.json)

    @staticmethod
    def _regulatory_bylaws() -> List[Dict[str, Any]]:
        if not _BYLAWS_JSON.exists():
            print(f"[RAG] Regulatory bylaws file not found: {_BYLAWS_JSON}")
            return []
        with open(_BYLAWS_JSON, encoding="utf-8") as f:
            docs = json.load(f)
        print(f"[RAG] Loaded {len(docs)} regulatory bylaw documents.")
        return docs


    # ChromaDB population

    def initialize_knowledge_base(self):
        if self.collection.count() > 0:
            print("[RAG] Knowledge base already initialized.")
            return
        for doc in self.knowledge_base:
            self.collection.add(
                ids=[doc["id"]],
                documents=[doc["content"]],
                metadatas=[{
                    "title": doc["title"],
                    "source": doc["source"],
                    "category": doc.get("category", "General"),
                }],
            )
        print(f"[RAG] Initialized knowledge base with {len(self.knowledge_base)} documents.")

    # Search

    def _keyword_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if len(w) > 2]
        scored = []
        for doc in self.knowledge_base:
            score = (
                sum(3 for kw in keywords if kw in doc["title"].lower()) +
                sum(1 for kw in keywords if kw in doc["content"].lower())
            )
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        return [
            {"title": d["title"], "content": d["content"],
             "source": d["source"], "category": d.get("category", "")}
            for _, d in scored[:top_k]
        ]

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        try:
            count = self.collection.count()
        except Exception:
            count = 0

        if count == 0:
            return self._keyword_search(query, top_k)

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, count),
            )
        except Exception:
            return self._keyword_search(query, top_k)

        if not results or not results["documents"] or not results["documents"][0]:
            return self._keyword_search(query, top_k)

        distances = results.get("distances", [[]])[0]
        documents = []
        for i, doc in enumerate(results["documents"][0]):
            documents.append({
                "title": results["metadatas"][0][i].get("title", ""),
                "content": doc,
                "source": results["metadatas"][0][i].get("source", ""),
                "category": results["metadatas"][0][i].get("category", ""),
                "distance": distances[i] if i < len(distances) else 1.0,
            })
        return documents


rag = TorontoBylawRAG()
