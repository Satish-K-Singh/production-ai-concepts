"""Evaluates and compares BM25, semantic, and hybrid search strategies."""

import re
import numpy as np
from rank_bm25 import BM25Okapi
from shop_common import FAQ_DOCUMENTS, embed_text

HYBRID_ALPHA = 0.5
TOP_K = 3

TEST_QUERIES = [
    (
        "keyword-heavy", 
        "Where can I find the PDF with furniture assembly instructions?"
    ),
    (
        "semantic paraphrase", 
        "The thing I bought doesn't work anymore, what are my options for "
        "getting reimbursed?"
    ),
    (
        "mixed", 
        "Can I still return my opened electronics if it's been 40 days?"
    ),
]

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "i", "you", "your", "yours", "my", "mine", "we", "our", "us", "they",
    "their", "it", "its", "this", "that", "these", "those", "of", "to",
    "for", "in", "on", "at", "by", "with", "and", "or", "but", "if", "so",
    "do", "does", "did", "doesnt", "didnt", "havent", "hasnt", "can",
    "could", "will", "would", "should", "what", "who", "which", "how",
    "where", "when", "why", "not", "no", "get", "getting", "got", "as",
}

def tokenize(text):
    """Tokenizes text into a list of lowercase words, removing stopwords.

    Args:
        text: The input string to tokenize.

    Returns:
        A list of cleaned, lowercase string tokens.
    """
    # Remove punctuation and split by whitespace
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS]

# Module-level data initialization for the mock search engine corpus
DOC_TEXTS = [doc["text"] for doc in FAQ_DOCUMENTS]
DOC_IDS = [doc["id"] for doc in FAQ_DOCUMENTS]
BM25 = BM25Okapi([tokenize(doc) for doc in DOC_TEXTS])
DOC_EMBEDDINGS = embed_text(DOC_TEXTS)

def normalize(scores:np.ndarray) -> np.ndarray:
    """Normalizes the input scores to a range between 0.0 and 1.0.

    Args:
        scores: A 1D numpy array of raw scores.

    Returns:
        A 1D numpy array of normalized scores. Returns an array of zeros 
        if all input scores are identical.
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score - min_score < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)

def bm25_scores(query:str) -> np.ndarray:
    """Computes BM25 keyword scores for a query against the document corpus.

    Args:
        query: The raw input search string.

    Returns:
        A 1D numpy array of float64 scores.
    """
    tokenized_query = tokenize(query)
    return np.array(BM25.get_scores(tokenized_query), dtype="float64")

def semantic_scores(query:str) -> np.ndarray:
    """Computes cosine similarity semantic scores for a query.

    Args:
        query: The raw input search string.

    Returns:
        A 1D numpy array of float32/float64 similarity scores based on 
        the embeddings.
    """
    query_embedding = embed_text([query])[0]
    return DOC_EMBEDDINGS @ query_embedding

def hybrid_scores(query:str, alpha:float=HYBRID_ALPHA) -> np.ndarray:
    """Computes hybrid scores by combining normalized BM25 and semantic scores.

    Args:
        query: The raw input search string.
        alpha: The weighting factor (0.0 to 1.0). 1.0 relies entirely on BM25, 
            while 0.0 relies entirely on semantic search.

    Returns:
        A 1D numpy array of the blended scores.
    """
    bm25 = bm25_scores(query)
    semantic = semantic_scores(query)
    return alpha * normalize(bm25) + (1 - alpha) * normalize(semantic)

def reciprocal_rank(query: str, k: int = 60) -> float:
    """Computes the Reciprocal Rank Fusion (RRF) scores for the given query.

    Args:
        query: The raw input search string.
        k: The smoothing constant for rank fusion, typically set between 
            50 and 60.

    Returns:
        A 1D numpy array of float scores representing the fused rankings.
    """
    semantic_rank = np.argsort(-semantic_scores(query)).argsort()
    bm25_rank = np.argsort(-bm25_scores(query)).argsort()
    return 1.0 / (k + semantic_rank + 1) + 1.0 / (k + bm25_rank + 1)


def print_ranking(label: str, scores: np.ndarray, top_k: int = TOP_K) -> None:
    """Sorts and prints the top k documents based on their scores.

    Args:
        label: A descriptive label for the ranking method being used.
        scores: A 1D numpy array of document scores.
        top_k: The number of top results to print.
    """
    order = np.argsort(-scores)[:top_k]
    print(f"  {label}:")
    for rank, idx in enumerate(order, start=1):
        print(f"    {rank}. [{DOC_IDS[idx]}] (score={scores[idx]:.4f}) {DOC_TEXTS[idx]}")

def main() -> None:
    """Evaluates keyword, semantic, and hybrid search methods on test queries."""
    separator = "=" * 80
    sub_separator = "-" * 80

    for query_type, query in TEST_QUERIES:
        print(separator)
        print(f"QUERY ({query_type}): {query}")
        print(sub_separator)
        
        # Run and print each search strategy
        print_ranking("KEYWORD ONLY (BM25)", bm25_scores(query))
        print_ranking("SEMANTIC ONLY (Cosine)", semantic_scores(query))
        print_ranking(
            f"HYBRID (Weighted Sum, alpha={HYBRID_ALPHA})", 
            hybrid_scores(query)
        )
        print_ranking(
            "HYBRID (Reciprocal Rank Fusion)", 
            reciprocal_rank(query)
        )
 
if __name__ == "__main__":
    main()

