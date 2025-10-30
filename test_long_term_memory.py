import uuid
from typing import Dict, Any, List, Optional

# ====================================================================
# MOCK DEPENDENCIES (simulate ChromaDB behavior)
# ====================================================================

class MockEmbeddingFunction:
    """Mocks a Gemini-compatible embedding function."""
    def __init__(self, model_name: str):
        pass

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return [[1.0] * 1536 for _ in texts]

class MockCollection:
    """Mocks Chroma Collection API methods."""
    def __init__(self, name, embedding_function):
        self.name = name
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embedding_function = embedding_function

    def add(self, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        for i, content in enumerate(documents):
            doc_id = ids[i] if ids and ids[i] else str(uuid.uuid4())
            metadata = metadatas[i] if metadatas and metadatas[i] else {}
            self.documents[doc_id] = {
                "document": content,
                "metadata": metadata,
                "embedding": self.embedding_function([content])[0]
            }

    def query(self, query_texts: List[str], n_results: int = 5) -> Dict[str, Any]:
        """Simulates retrieval using keyword matching (all query words must appear)."""
        query_words = query_texts[0].lower().split()
        results_docs = []
        results_metadatas = []

        for doc_id, data in self.documents.items():
            doc_lower = data['document'].lower()
            if all(word in doc_lower for word in query_words):
                results_docs.append(data['document'])
                results_metadatas.append(data['metadata'])
                if len(results_docs) >= n_results:
                    break

        return {
            "ids": [[]],
            "documents": [results_docs],
            "metadatas": [results_metadatas]
        }

    def get(self, include: List[str]):
        return {"documents": [d['document'] for d in self.documents.values()]}

    def delete(self, ids: List[str]):
        for doc_id in ids:
            if doc_id in self.documents:
                del self.documents[doc_id]

class MockClient:
    """Mocks Chroma Client."""
    def __init__(self, path: str = None):
        self.collections: Dict[str, MockCollection] = {}

    def list_collections(self):
        return [c for c in self.collections.values()]

    def get_collection(self, name: str):
        return self.collections.get(name)

    def create_collection(self, name: str, embedding_function):
        collection = MockCollection(name, embedding_function)
        self.collections[name] = collection
        return collection

class VectorStore:
    """Interface to the mock vector store."""
    def __init__(self, persist_directory: str = "memory/chroma_db"):
        self.client = MockClient(path=persist_directory)
        self.embedding_fn = MockEmbeddingFunction(model_name="gemini-embedding-001")
        self.collection_name = "ai_agent_memory"

        if self.collection_name in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(name=self.collection_name)
        else:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )

    def add_memory(self, content: str, metadata: Dict = None):
        self.collection.add(documents=[content], metadatas=[metadata or {}])

    def retrieve_memory(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        return self.collection.query(query_texts=[query], n_results=top_k)

    def list_memories(self):
        return self.collection.get(include=["documents", "metadatas", "ids"])

    def delete_memory(self, memory_id: str):
        self.collection.delete(ids=[memory_id])

# ====================================================================
# LONGTERM MEMORY CLASS
# ====================================================================

class LongTermMemory:
    def __init__(self):
        self.store = VectorStore()
        self.initialize_strategic_knowledge()

    def initialize_strategic_knowledge(self):
        self.store.add_memory(
            content="Key performance indicators (KPIs) for airlines include On-Time Performance (OTP), Revenue Per Available Seat Mile (RASM), and Load Factor.",
            metadata={"source": "Internal Policy 2024", "id": "kpi_doc"}
        )
        self.store.add_memory(
            content="Strategic goal: Reduce passenger delay minutes by 15% in the next quarter via improved ground operations.",
            metadata={"source": "Executive Mandate", "id": "strategic_goal"}
        )
        self.store.add_memory(
            content="The standard procedure for reporting major anomalies (>$100k cost variance) requires immediate escalation to the Financial Planning team.",
            metadata={"source": "SOP Manual", "id": "anomaly_sop"}
        )

    def remember(self, key: str, text: str):
        self.store.add_memory(
            content=text,
            metadata={"source": "Agent Insight", "id": key}
        )

    def recall(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        results = self.store.retrieve_memory(query, top_k)
        context = []
        if results.get('documents') and results['documents']:
            for doc_list in results['documents']:
                context.extend(doc_list)
        return {"context": context, "raw_results": results}

# ====================================================================
# TEST EXECUTION
# ====================================================================

def test_long_term_memory():
    print("--- Starting LongTermMemory Tests ---")
    memory = LongTermMemory()

    # Test 1: Initialization
    initial_docs = memory.store.collection.get(include=["documents"])['documents']
    assert len(initial_docs) == 3, f"Expected 3 documents, got {len(initial_docs)}"
    print(f"  ✅ Initial documents count correct: {len(initial_docs)}")

    # Test 2: Remember a new fact
    new_key = "new_flight_data_Q3"
    new_text = "New data shows flight cancellations spiked by 20% in the last quarter due to maintenance issues."
    memory.remember(new_key, new_text)
    docs_after_add = memory.store.collection.get(include=["documents"])['documents']
    assert len(docs_after_add) == 4, "Expected 4 documents after adding new insight"
    print("  ✅ New fact successfully added.")

    # Test 3: Recall (RAG Simulation)
    query_goal = "reduce passenger delay minutes"
    results_goal = memory.recall(query_goal, top_k=1)
    assert len(results_goal['context']) > 0, "Recall failed for strategic goal"
    assert any("Reduce passenger delay minutes" in doc for doc in results_goal['context']), \
        "Recall retrieved incorrect context"
    print("  ✅ Recall of strategic goal successful.")

    query_new = "flight cancellations maintenance issues"
    results_new = memory.recall(query_new, top_k=1)
    assert len(results_new['context']) > 0, "Recall failed for new insight"
    assert any("maintenance issues" in doc for doc in results_new['context']), \
        "Recall retrieved incorrect context"
    print("  ✅ Recall of new insight successful.")

    print("\n--- All LongTermMemory Tests Passed! ---")

if __name__ == "__main__":
    test_long_term_memory()
