class VectorBackend:
    """Contract every backend (pgvector, Qdrant, ...) implements.

    build_index config examples:
      {"type": "hnsw", "m": 16, "ef_construction": 64}
      {"type": "ivfflat", "lists": 100}
      {"type": "none"}                  # no index, exact search
    search params examples: {"ef_search": 40} or {"probes": 10}

    The harness asks for top_k + 1 results and drops the query's own id.
    Distance is L2 on all-MiniLM-L6-v2 embeddings; ids are 1..100000.
    """
    name = "base"

    def build_index(self, config: dict) -> float:
        """Build the index and return build time in seconds."""
        raise NotImplementedError

    def search(self, vector: list, top_k: int, params: dict = None) -> list:
        """Return [{"id": int, "distance": float}, ...], nearest first."""
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def close(self):
        pass