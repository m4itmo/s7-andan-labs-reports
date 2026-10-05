"""Reusable functions for the RAG-publications graph analysis."""
from __future__ import annotations

from collections import Counter
import re
from itertools import combinations

import networkx as nx

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "with", "we", "our",
    "using", "use", "based", "new", "via", "can", "may", "their", "than", "such", "also", "not", "when", "over", "where", "which", "within", "but", "how", "all", "more", "only", "has", "these", "through", "while", "without", "both", "every",
    "large", "language", "paper", "approach", "method", "methods",
}


def _lemma(token: str) -> str:
    """Small deterministic lemmatizer sufficient for transparent keyword cleanup."""
    irregular = {"retrieve": "retrieval", "retrieves": "retrieval", "retrieved": "retrieval", "models": "model", "documents": "document", "queries": "query", "corpus": "corpus"}
    if token in irregular:
        return irregular[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_terms(text: str) -> list[str]:
    """Lowercase, remove stopwords, normalize basic English word forms."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z-]{1,}", text.lower())
    return [_lemma(token) for token in tokens if token not in STOPWORDS and len(token) > 2]


def extract_keywords(text: str, vocabulary: set[str], max_keywords: int = 12) -> list[str]:
    """Select frequent normalized unigram terms constrained by a corpus vocabulary."""
    counts = Counter(token for token in normalize_terms(text) if token in vocabulary)
    return [term for term, _ in counts.most_common(max_keywords)]


def build_keyword_graph(keyword_lists: list[list[str]]) -> nx.Graph:
    """Build an undirected weighted co-occurrence graph."""
    graph = nx.Graph()
    for terms in keyword_lists:
        unique_terms = sorted(set(terms))
        graph.add_nodes_from(unique_terms)
        for left, right in combinations(unique_terms, 2):
            if graph.has_edge(left, right):
                graph[left][right]["weight"] += 1
            else:
                graph.add_edge(left, right, weight=1)
    return graph


def build_publication_graph(records: list[dict]) -> nx.Graph:
    """Link publications by shared keywords; edge weight is shared-keyword count."""
    graph = nx.Graph()
    for record in records:
        graph.add_node(record["id"], title=record.get("title", ""), keywords=record["keywords"])
    for left, right in combinations(records, 2):
        shared = sorted(set(left["keywords"]) & set(right["keywords"]))
        if shared:
            graph.add_edge(left["id"], right["id"], weight=len(shared), shared_keywords=shared)
    return graph


def nearest_publications(graph: nx.Graph, publication_id: str, top_k: int = 5) -> list[dict]:
    """Return graph neighbors ranked by the number of shared keywords."""
    rows = []
    for neighbor, attrs in graph[publication_id].items():
        rows.append({"id": neighbor, "shared_keywords": attrs["weight"], "keywords": attrs["shared_keywords"]})
    return sorted(rows, key=lambda row: (-row["shared_keywords"], row["id"]))[:top_k]
