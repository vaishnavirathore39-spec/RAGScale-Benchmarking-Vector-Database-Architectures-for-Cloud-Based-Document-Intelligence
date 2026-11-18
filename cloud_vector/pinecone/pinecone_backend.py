
import csv
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, DenseVectorQuery

from common.backend_interface import VectorBackend


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

ENV_FILE = Path(
    os.getenv(
        "RAGSCALE_ENV_FILE",
        str(BASE_DIR / ".env")
    )
)

load_dotenv(ENV_FILE)

EXPECTED_DIMENSION = 384


# ==========================================
# 2. PINECONE BACKEND
# ==========================================

class PineconeBackend(VectorBackend):

    name = "pinecone"

    def __init__(
        self,
        namespace="arxiv",
        mapping_file=None
    ):

        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")

        if not api_key or not index_name:
            raise ValueError(
                "PINECONE_API_KEY or PINECONE_INDEX_NAME "
                "missing from environment"
            )

        self.namespace = namespace

        self.mapping_file = (
            Path(mapping_file)
            if mapping_file
            else Path(__file__).resolve().parent / "id_mapping.csv"
        )

        if not self.mapping_file.exists():
            raise FileNotFoundError(
                f"ID mapping not found: {self.mapping_file}"
            )

        # Load ID mapping
        self.arxiv_to_numeric = {}
        self.numeric_to_arxiv = {}

        with open(
            self.mapping_file,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                numeric_id = int(row["numeric_id"])
                arxiv_id = str(row["arxiv_id"])

                if numeric_id in self.numeric_to_arxiv:
                    raise ValueError(
                        f"Duplicate numeric ID: {numeric_id}"
                    )

                if arxiv_id in self.arxiv_to_numeric:
                    raise ValueError(
                        f"Duplicate arXiv ID: {arxiv_id}"
                    )

                self.numeric_to_arxiv[numeric_id] = arxiv_id
                self.arxiv_to_numeric[arxiv_id] = numeric_id

        # Connect to existing Pinecone index
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

        self.index_name = index_name

        print("Connected to Pinecone:", self.index_name)
        print("Namespace:", self.namespace)
        print("Mapping records:", len(self.numeric_to_arxiv))


    # ==========================================
    # 3. BUILD INDEX
    # ==========================================

    def build_index(self, config):

        """
        Pinecone manages its indexing internally.
        PostgreSQL HNSW/IVFFlat settings are not applied.
        """

        start = time.perf_counter()

        if config and config.get("type") not in (
            "none",
            "managed"
        ):

            print(
                "Pinecone manages indexing internally. "
                "PostgreSQL index settings are not applicable."
            )

        return time.perf_counter() - start


    # ==========================================
    # 4. VECTOR SEARCH
    # ==========================================

    def search(self, vector, top_k, params=None):

        """
        Search Pinecone and return results in common format:

        [
            {"id": integer, "distance": float}
        ]
        """

        if len(vector) != EXPECTED_DIMENSION:
            raise ValueError(
                f"Expected {EXPECTED_DIMENSION} dimensions, "
                f"received {len(vector)}"
            )

        query_vector = [
            float(value) for value in vector
        ]

        response = self.index.documents.search(
            namespace=self.namespace,
            top_k=int(top_k),
            score_by=[
                DenseVectorQuery(
                    field="embedding",
                    values=query_vector
                )
            ]
        )

        results = []

        for match in response.matches:

            arxiv_id = str(match.id)

            # Convert Pinecone string ID to numeric ID
            numeric_id = self.arxiv_to_numeric.get(arxiv_id)

            if numeric_id is None:
                continue

            # Pinecone cosine similarity score
            cosine_score = float(match.score)

            cosine_score = max(
                -1.0,
                min(1.0, cosine_score)
            )

            # Convert cosine similarity to L2 distance.
            # Valid for unit-normalized vectors.
            distance = (
                2.0 * (1.0 - cosine_score)
            ) ** 0.5

            results.append({
                "id": numeric_id,
                "distance": distance
            })

        return results


    # ==========================================
    # 5. VECTOR COUNT
    # ==========================================

    def count(self):

        """Return existing namespace vector count."""

        stats = self.index.describe_index_stats()

        namespace_stats = stats.namespaces.get(
            self.namespace
        )

        if namespace_stats is None:
            return 0

        return namespace_stats.vector_count


    # ==========================================
    # 6. CLOSE
    # ==========================================

    def close(self):

        """No persistent database connection to close."""

        pass
