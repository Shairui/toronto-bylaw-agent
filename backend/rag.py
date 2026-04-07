"""RAG (Retrieval-Augmented Generation) knowledge base for Toronto bylaws."""
import json
import os
from typing import List, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
from backend.config import CHROMA_DB_PATH, KNOWLEDGE_BASE_PATH


class TorontoBylawRAG:
    """RAG system for Toronto bylaw knowledge base."""
    
    def __init__(self):
        """Initialize RAG system with ChromaDB."""
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_DB_PATH,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name="toronto_bylaws",
            metadata={"hnsw:space": "cosine"}
        )
        self.knowledge_base = self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        """Load knowledge base from JSON file."""
        if os.path.exists(KNOWLEDGE_BASE_PATH):
            with open(KNOWLEDGE_BASE_PATH, 'r') as f:
                return json.load(f)
        return self._create_default_knowledge_base()
    
    def _create_default_knowledge_base(self) -> List[Dict[str, Any]]:
        """Create default Toronto bylaw knowledge base."""
        return [
            {
                "id": "bylaw_001",
                "title": "Toronto Municipal Code - Chapter 349 (Zoning)",
                "content": "Chapter 349 of the Toronto Municipal Code covers zoning regulations. Key points: Residential zones (R1-R4), Commercial zones (C1-C4), Industrial zones (I1-I2). Building heights, setbacks, and lot coverage vary by zone.",
                "source": "Toronto Municipal Code",
                "category": "Zoning"
            },
            {
                "id": "bylaw_002",
                "title": "Building Permit Requirements",
                "content": "Building permits are required for: New construction, Major renovations (>25% of property value), Structural changes, HVAC/electrical/plumbing upgrades. Exemptions: Minor repairs, painting, landscaping.",
                "source": "Toronto Building Department",
                "category": "Permits"
            },
            {
                "id": "bylaw_003",
                "title": "Waste Collection Schedule",
                "content": "Toronto provides weekly garbage, recycling, and organic waste collection. Collection days vary by neighborhood. Bins must be placed at curb by 6 AM and removed by 10 PM same day.",
                "source": "Toronto Waste Management",
                "category": "Waste"
            },
            {
                "id": "bylaw_004",
                "title": "Hazard Reporting (311)",
                "content": "Report hazards like potholes, fallen trees, debris via Toronto 311. Available 24/7. Provide location, hazard type, and description. Response time varies by severity.",
                "source": "Toronto 311 Service",
                "category": "Hazards"
            },
            {
                "id": "bylaw_005",
                "title": "Property Standards",
                "content": "Property must be maintained in good condition. Violations include: Broken windows, deteriorated exterior, overgrown vegetation, abandoned vehicles. Fines range from $500-$5000.",
                "source": "Toronto Municipal Code",
                "category": "Property"
            }
        ]
    
    def initialize_knowledge_base(self):
        """Initialize ChromaDB with knowledge base documents."""
        if self.collection.count() > 0:
            print("[RAG] Knowledge base already initialized")
            return
        
        for doc in self.knowledge_base:
            self.collection.add(
                ids=[doc["id"]],
                documents=[doc["content"]],
                metadatas=[{
                    "title": doc["title"],
                    "source": doc["source"],
                    "category": doc.get("category", "General")
                }]
            )
        print(f"[RAG] Initialized knowledge base with {len(self.knowledge_base)} documents")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if not results or not results["documents"] or not results["documents"][0]:
            return []
        
        documents = []
        for i, doc in enumerate(results["documents"][0]):
            documents.append({
                "title": results["metadatas"][0][i].get("title", ""),
                "content": doc,
                "source": results["metadatas"][0][i].get("source", ""),
                "category": results["metadatas"][0][i].get("category", "")
            })
        
        return documents


# Global RAG instance
rag = TorontoBylawRAG()
