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

    # Regulatory bylaw texts (no external source file exists for these)

    @staticmethod
    def _regulatory_bylaws() -> List[Dict[str, Any]]:
        return [
            {
                "id": "bylaw_permits_required",
                "title": "When a Building Permit Is Required",
                "content": "Building permits are required for: new construction over 10 m²; additions; demolition; structural alterations; HVAC, plumbing, or drain changes; decks more than 600 mm above grade; pools and hot tubs; fireplaces. Exemptions: painting, flooring, cabinet installation, minor cosmetic repairs.",
                "source": "Toronto Municipal Code Chapter 363; Ontario Building Code",
                "category": "Permits",
            },
            {
                "id": "bylaw_permits_apply",
                "title": "How to Apply for a Building Permit",
                "content": "Apply at toronto.ca/building. Required: application form, site plan, construction drawings (architect/engineer seal for complex projects). Review times: 10–20 business days for small residential; 30–60 for larger projects. Fee: $13.73 per $1,000 of construction value (minimum $196).",
                "source": "Toronto Municipal Code Chapter 363; toronto.ca/building",
                "category": "Permits",
            },
            {
                "id": "bylaw_deck",
                "title": "Deck Permits and Requirements",
                "content": "A building permit is required for decks more than 600 mm above grade. Guard rails (min 1.07 m high) required above 600 mm. Footings must extend to frost depth (~1.2 m). Apply at toronto.ca/building or call 311.",
                "source": "Ontario Building Code; Toronto Zoning By-law 569-2013",
                "category": "Permits",
            },
            {
                "id": "bylaw_pool",
                "title": "Swimming Pool Permit and Enclosure",
                "content": "A building permit is required for any in-ground or above-ground pool. A fence at least 1.2 m high with self-latching gate must surround the pool. Hot tubs follow the same rules. Fine up to $100,000 for non-compliance.",
                "source": "Toronto Municipal Code Chapter 447; Ontario Building Code",
                "category": "Permits",
            },
            {
                "id": "bylaw_zoning_overview",
                "title": "Toronto Zoning By-law 569-2013 Overview",
                "content": "Zoning By-law 569-2013 divides Toronto into residential (R, RT, RA), commercial (C), employment (E), and mixed-use (CR) zones. Each zone sets rules for permitted uses, density, setbacks, lot coverage, building height, and parking. Front yard setback: 6 m. Rear yard: 7.5 m. Max residential height: 10 m (R zones).",
                "source": "Toronto Zoning By-law 569-2013",
                "category": "Zoning",
            },
            {
                "id": "bylaw_secondary_suite",
                "title": "Secondary Suites and Garden Suites",
                "content": "Secondary suites (basement apartments) are permitted as-of-right in all residential zones (Bill 23, 2022). Garden suites (backyard dwellings) are also permitted — max 60 m² and 6 m height. A building permit is required for both.",
                "source": "Toronto Zoning By-law 569-2013, Section 150; City of Toronto 2022",
                "category": "Zoning",
            },
            {
                "id": "bylaw_noise_construction",
                "title": "Construction Noise — Permitted Hours",
                "content": "Construction noise is permitted Monday–Friday 7 AM–7 PM and Saturday 9 AM–7 PM. Construction noise is prohibited on Sundays and statutory holidays. Report violations to 311.",
                "source": "Toronto Municipal Code Chapter 591 (Noise)",
                "category": "Noise",
            },
            {
                "id": "bylaw_noise_residential",
                "title": "Residential Noise — Quiet Hours",
                "content": "Quiet hours: 11 PM–7 AM on weekdays; 11 PM–9 AM on weekends. Amplified music audible from neighbouring properties after these hours is a violation. Fines: $500–$5,000. Report to Toronto Police (non-emergency) at 416-808-2222 or call 311.",
                "source": "Toronto Municipal Code Chapter 591 (Noise)",
                "category": "Noise",
            },
            {
                "id": "bylaw_parking_onstreet",
                "title": "On-Street Parking Rules",
                "content": "Parking is prohibited within 9 m of an intersection, and on arterial roads during rush hours (7–9 AM, 4–6 PM weekdays). Maximum parking time in most residential areas is 3 hours. Dispute tickets within 15 days at toronto.ca/parking-tickets.",
                "source": "Toronto Municipal Code Chapter 950 (Traffic and Parking)",
                "category": "Parking",
            },
            {
                "id": "bylaw_parking_winter",
                "title": "Overnight Winter Parking Ban",
                "content": "Parking on most city streets is prohibited from 12 AM to 7 AM, November 15 to April 1. Violation: $100 fine; vehicle may be towed. Residents can apply for a Residential Permit Parking permit.",
                "source": "Toronto Municipal Code Chapter 950",
                "category": "Parking",
            },
            {
                "id": "bylaw_property_exterior",
                "title": "Exterior Property Maintenance Standards",
                "content": "Under Chapter 629, all property owners must maintain walls, roofs, foundations, windows, eavestroughs, and driveways in good repair. Exterior surfaces must be free from peeling paint and graffiti. File complaints via 311.",
                "source": "Toronto Municipal Code Chapter 629 (Property Standards)",
                "category": "Property",
            },
            {
                "id": "bylaw_property_interior",
                "title": "Interior Property Standards (Rental Units)",
                "content": "Heating must maintain 21°C from September 15 to June 1. Hot water minimum 43°C. Kitchens need working stove and fridge; bathrooms need toilet, sink, and tub or shower. Landlords are responsible for maintenance. Tenants may apply to the Landlord and Tenant Board.",
                "source": "Toronto Municipal Code Chapter 629; Residential Tenancies Act",
                "category": "Property",
            },
            {
                "id": "bylaw_fence",
                "title": "Fence By-law",
                "content": "Front yard fences: maximum 1.0 m. Rear and side yard fences: maximum 2.0 m. Barbed wire and electrified fences are prohibited in residential zones. No permit required for fences under 2.0 m.",
                "source": "Toronto Municipal Code Chapter 447; Line Fences Act",
                "category": "Property",
            },
            {
                "id": "bylaw_graffiti",
                "title": "Graffiti By-law",
                "content": "Property owners must remove graffiti within 30 days of it appearing (5 business days after City notice). The City offers free first-time removal on properties facing public right-of-way. Report at toronto.ca/graffiti or 311. Fine for non-removal: $360.",
                "source": "Toronto Municipal Code Chapter 485 (Graffiti)",
                "category": "Property",
            },
            {
                "id": "bylaw_trees_private",
                "title": "Private Tree Protection",
                "content": "Trees on private property with trunk diameter ≥ 30 cm are protected. A permit is required to injure or remove them. Unauthorized removal: fine of $500–$100,000 per tree. Apply at toronto.ca/trees.",
                "source": "Toronto Municipal Code Chapter 813 (Trees)",
                "category": "Trees",
            },
            {
                "id": "bylaw_trees_street",
                "title": "Street Tree Protection",
                "content": "City-owned street trees may not be damaged or removed without written Urban Forestry approval. Protective fencing is required during any construction near street trees. To request a new street tree, contact 311.",
                "source": "Toronto Municipal Code Chapter 813; Urban Forestry",
                "category": "Trees",
            },
            {
                "id": "bylaw_parks",
                "title": "Park Use Rules",
                "content": "Parks are open 5:30 AM to midnight. Alcohol requires a special event permit. Overnight camping is prohibited. Groups over 25 people need a Park Use Permit (apply at toronto.ca/permits, 4 weeks in advance).",
                "source": "Toronto Municipal Code Chapter 608 (Parks)",
                "category": "Parks",
            },
            {
                "id": "bylaw_dogs",
                "title": "Dogs in Parks — Off-Leash Areas",
                "content": "Dogs must be on-leash except in designated off-leash areas (60+ across Toronto). Off-leash fine: $365. Dogs are prohibited from beaches, wading pools, and playgrounds. Dogs must be licensed annually: $50 (spayed/neutered) or $90 (intact).",
                "source": "Toronto Municipal Code Chapter 608; Chapter 349 (Animals)",
                "category": "Parks",
            },
            {
                "id": "bylaw_business_licensing",
                "title": "Business Licensing in Toronto",
                "content": "Many business types require a Municipal Licence from ML&S: restaurants, taxis/rideshare (Uber, Lyft), rooming houses, pet shops, tow trucks. Apply at toronto.ca/business. Operating without a required licence is an offence.",
                "source": "Toronto Municipal Code Chapter 545 (Licensing)",
                "category": "Business",
            },
            {
                "id": "bylaw_shortterm_rental",
                "title": "Short-Term Rentals (Airbnb)",
                "content": "Short-term rentals are only permitted in a host's principal residence. Entire-home rentals: maximum 180 nights per year. Room rentals: unlimited. Operators must register with the City and collect the Municipal Accommodation Tax (4%).",
                "source": "Toronto Municipal Code Chapter 547 (Short-Term Rentals)",
                "category": "Zoning",
            },
            {
                "id": "bylaw_311",
                "title": "Toronto 311 — Service Requests and Bylaw Complaints",
                "content": "311 is available 24/7 by phone or at toronto.ca/311. Use it to report potholes, fallen trees, graffiti, illegal dumping, noise, property standards violations, and parking infractions. Track service request status online with your confirmation number.",
                "source": "City of Toronto — toronto.ca/311",
                "category": "General",
            },
        ]


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
