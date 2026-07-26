"""Benchmarks FAISS indexing and search performance using clustered data.

This module provides utilities to generate synthetic clustered vectors,
build FAISS Flat and HNSW indexes, measure search latency, and calculate
recall@k against ground-truth data.
"""
import time
import faiss
import numpy as np
from shop_common import EMBEDDING_DIM


CORPUS_SIZE = 20_000
NUM_QUERIES = 200
NUM_CLUSTERS = 200
CLUSTER_NOISE_STD = 0.08
TOP_K = 10
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
EF_SEARCH_SETTINGS = [8, 16, 32, 64, 128, 256]

# Module-level random number generator for reproducibility
rng = np.random.default_rng(seed=42)

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalizes a batch of vectors to the unit sphere using L2 norm.

    Args:
        vectors: A 2D numpy array of vectors to be normalized.

    Returns:
        A 2D numpy array of L2-normalized vectors.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Prevent division by zero if a zero-vector exists
    norms = np.where(norms == 0, 1e-10, norms)
    return vectors / norms

def masked_clustered(n: int, dim: int, centers: np.ndarray) -> np.ndarray:
    """Samples vectors scattered around cluster centers and normalizes them.

    Args:
        n: The number of vectors to generate.
        dim: The dimensionality of the vectors.
        centers: A 2D numpy array containing the cluster center vectors.

    Returns:
        A 2D numpy array of shape (n, dim) containing the generated
        unit-normalized vectors, cast to float32.
    """
    center_indices = rng.integers(0, len(centers), size=n)
    noise = (
        rng.standard_normal((n, dim)).astype("float32") * CLUSTER_NOISE_STD
    )
    vectors = centers[center_indices] + noise
    return normalize_vectors(vectors).astype("float32")

# FAISS Flat Index
def build_flat_index(vectors: np.ndarray) -> tuple[faiss.IndexFlatL2, float]:
    """Builds a FAISS Flat L2 index from the provided vectors.

    Args:
        vectors: A 2D numpy array of float32 vectors to index.

    Returns:
        A tuple containing the constructed FAISS IndexFlatL2 object and 
        the time taken to build it (in seconds).
    """
    start = time.perf_counter()
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(vectors)
    build_time = time.perf_counter() - start
    return index, build_time

#FAISS HNSW Index
def build_hnsw_index(vectors: np.ndarray) -> tuple[faiss.IndexHNSWFlat, float]:
    """Builds a FAISS HNSW Flat index from the provided vectors.

    Args:
        vectors: A 2D numpy array of float32 vectors to index.

    Returns:
        A tuple containing the constructed FAISS IndexHNSWFlat object and 
        the time taken to build it (in seconds).
    """
    start = time.perf_counter()
    index = faiss.IndexHNSWFlat(EMBEDDING_DIM, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(vectors)
    build_time = time.perf_counter() - start
    return index, build_time

def time_search(index, queries:np.ndarray, top_k:int) -> float:
    """Searches a FAISS index and measures the total query time.

    Args:
        index: The FAISS index object to search against.
        queries: A 2D numpy array of query vectors.
        top_k: The number of nearest neighbors to retrieve per query.

    Returns:
        A tuple containing a 2D numpy array of the nearest neighbor indices 
        and the search duration (in seconds).
    """
    start = time.perf_counter()
    _, neighbours = index.search(queries, top_k)
    search_time = time.perf_counter() - start
    return neighbours, search_time

def recall_k(ground_truth: np.ndarray, candidate: np.array) -> float:
    """Computes the recall@k metric for a set of queries.

    Args:
        ground_truth: A 2D numpy array where each row contains the true 
            nearest neighbor indices for a single query.
        candidate: A 2D numpy array where each row contains the algorithm's 
            predicted nearest neighbor indices for that query.

    Returns:
        The average recall across all queries, as a float between 0.0 and 1.0.
    """
    correct = 0
    for gt, cand in zip(ground_truth, candidate):
        correct += len(set(gt).intersection(set(cand)))

    total_targets = ground_truth.shape[0] * ground_truth.shape[1]

    return correct / total_targets

def main() -> None:
    """Runs the FAISS benchmark comparing Flat L2 and HNSW indices."""
    separator = "=" * 80
    sub_separator = "-" * 80

    print(separator)
    print(
        f"Building synthetic corpus: {CORPUS_SIZE} vectors x "
        f"{EMBEDDING_DIM} dims\n({NUM_CLUSTERS} topic clusters, "
        "mimicking real embedding structure)"
    )
    
    # 1. Data Generation
    raw_centers = rng.standard_normal((NUM_CLUSTERS, EMBEDDING_DIM))
    cluster_centers = normalize_vectors(raw_centers.astype(np.float32))
    
    corpus = masked_clustered(CORPUS_SIZE, EMBEDDING_DIM, cluster_centers)
    queries = masked_clustered(NUM_QUERIES, EMBEDDING_DIM, cluster_centers)

    # 2. Index Building
    flat_index, flat_build_time = build_flat_index(corpus)
    hnsw_index, hnsw_build_time = build_hnsw_index(corpus)

    print(f"Flat L2 build time:  {flat_build_time:.3f}s")
    print(
        f"HNSW build time:     {hnsw_build_time:.3f}s "
        f"(M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION})"
    )

    # 3. Ground Truth (Exact Search)
    ground_truth, flat_query_time = time_search(flat_index, queries, TOP_K)
    flat_avg_latency_ms = (flat_query_time / NUM_QUERIES) * 1000
    
    print(
        f"\nFlat L2 (ground truth): {NUM_QUERIES} queries in "
        f"{flat_query_time:.4f}s\n({flat_avg_latency_ms:.4f} ms/query avg), "
        f"recall@{TOP_K}=1.0000 (exact)"
    )

    # 4. Approximate Search (HNSW) Evaluation
    print(f"\n{sub_separator}")
    print(
        f"{'efSearch':>10} | {'avg latency (ms)':>18} | "
        f"{'recall@' + str(TOP_K):>12} | {'speedup vs Flat':>16}"
    )
    print(sub_separator)
    
    for ef_search in EF_SEARCH_SETTINGS:
        hnsw_index.hnsw.efSearch = ef_search
        hnsw_neighbors, hnsw_query_time = time_search(
            hnsw_index, queries, TOP_K
        )
        
        avg_latency_ms = (hnsw_query_time / NUM_QUERIES) * 1000
        recall = recall_k(ground_truth, hnsw_neighbors)
        speedup = flat_query_time / hnsw_query_time
        
        print(
            f"{ef_search:>10} | {avg_latency_ms:>18.4f} | "
            f"{recall:>12.4f} | {speedup:>15.2f}x"
        )
   

if __name__ == "__main__":
    main()

