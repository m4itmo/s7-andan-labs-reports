"""End-to-end Lab 1: graph analysis of RAG publications from arXiv."""
from __future__ import annotations

import csv
import json
import math
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from lab1_analysis import build_keyword_graph, build_publication_graph, nearest_publications, normalize_terms

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
QUERY = 'all:"retrieval augmented generation"'


def fetch_arxiv(target: int = 400, batch_size: int = 100) -> list[dict]:
    """Fetch unique RAG records, retaining only papers from 2020 onward."""
    records, seen = [], set()
    for start in range(0, target + batch_size, batch_size):
        params = urllib.parse.urlencode({"search_query": QUERY, "start": start, "max_results": batch_size, "sortBy": "submittedDate", "sortOrder": "descending"})
        request = urllib.request.Request(f"https://export.arxiv.org/api/query?{params}", headers={"User-Agent": "lab1-rag-graph/1.0 (educational)"})
        with urllib.request.urlopen(request, timeout=90) as response:
            root = ET.fromstring(response.read())
        entries = root.findall("atom:entry", NS)
        if not entries:
            break
        for entry in entries:
            published = entry.findtext("atom:published", default="", namespaces=NS)
            identifier = entry.findtext("atom:id", default="", namespaces=NS).rsplit("/", 1)[-1]
            if published[:4] < "2020" or identifier in seen:
                continue
            seen.add(identifier)
            records.append({
                "id": identifier,
                "published": published[:10],
                "title": " ".join(entry.findtext("atom:title", default="", namespaces=NS).split()),
                "abstract": " ".join(entry.findtext("atom:summary", default="", namespaces=NS).split()),
                "authors": "; ".join(a.findtext("atom:name", default="", namespaces=NS) for a in entry.findall("atom:author", NS)),
                "categories": "; ".join(c.attrib.get("term", "") for c in entry.findall("atom:category", NS)),
            })
            if len(records) >= target:
                return records
        time.sleep(3)
    return records


def main() -> None:
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    records = fetch_arxiv()
    if len(records) < 300:
        raise RuntimeError(f"arXiv returned only {len(records)} records; at least 300 are required")
    corpus = [f"{r['title']} {r['abstract']}" for r in records]
    all_tokens = [token for text in corpus for token in normalize_terms(text)]
    counts = Counter(all_tokens)
    # Corpus-specific stopwords are excluded after inspection of the top-frequency list.
    domain_stopwords = {"retrieval", "augmented", "generation", "rag", "model", "system", "task", "data", "information", "research", "results", "propose", "provide", "work"}
    def analyzer(text: str) -> list[str]:
        tokens = [t for t in normalize_terms(text) if t not in domain_stopwords]
        return tokens + [" ".join(tokens[i:i+n]) for n in (2, 3) for i in range(len(tokens) - n + 1)]
    vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=5, max_df=0.55, max_features=180, sublinear_tf=True)
    matrix = vectorizer.fit_transform(corpus)
    terms = np.array(vectorizer.get_feature_names_out())
    for row, record in enumerate(records):
        indices = matrix[row].toarray().ravel().argsort()[-10:][::-1]
        record["keywords"] = terms[indices].tolist()

    (DATA / "rag_arxiv_2020plus.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with (DATA / "rag_arxiv_2020plus.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)

    keyword_graph = build_keyword_graph([r["keywords"] for r in records])
    # Suppress single co-occurrences for a readable, stable semantic graph.
    keyword_graph.remove_edges_from([(a, b) for a, b, d in keyword_graph.edges(data=True) if d["weight"] < 3])
    keyword_graph.remove_nodes_from(list(nx.isolates(keyword_graph)))
    communities = list(nx.community.greedy_modularity_communities(keyword_graph, weight="weight"))
    modularity = nx.community.modularity(keyword_graph, communities, weight="weight")
    degree = nx.degree_centrality(keyword_graph)
    betweenness = nx.betweenness_centrality(keyword_graph, weight="weight")
    eigenvector = nx.eigenvector_centrality(keyword_graph, weight="weight", max_iter=2000)
    community_rows = []
    for number, community in enumerate(sorted(communities, key=len, reverse=True), 1):
        top = sorted(community, key=lambda t: degree[t], reverse=True)[:8]
        community_rows.append({"community": number, "size": len(community), "terms": ", ".join(top)})
    centrality_rows = [{"term": term, "degree": degree[term], "betweenness": betweenness[term], "eigenvector": eigenvector[term]} for term in keyword_graph]
    centrality_rows.sort(key=lambda r: r["eigenvector"], reverse=True)

    publication_graph = build_publication_graph(records)
    query_record = max(records, key=lambda r: len(publication_graph[r["id"]]))
    search_results = nearest_publications(publication_graph, query_record["id"], 5)
    for item in search_results:
        item["title"] = publication_graph.nodes[item["id"]]["title"]

    # EDA plots.
    years = Counter(r["published"][:4] for r in records)
    plt.figure(figsize=(8, 4)); plt.bar(sorted(years), [years[y] for y in sorted(years)], color="#3b82f6")
    plt.title("RAG publications in the arXiv sample by year"); plt.xlabel("Year"); plt.ylabel("Publications"); plt.tight_layout(); plt.savefig(FIGURES / "publications_by_year.png", dpi=160); plt.close()
    ranks = np.arange(1, min(300, len(counts)) + 1); frequencies = np.array([freq for _, freq in counts.most_common(len(ranks))])
    slope, intercept = np.polyfit(np.log(ranks), np.log(frequencies), 1)
    plt.figure(figsize=(6, 4)); plt.loglog(ranks, frequencies, "o", markersize=3, label="observed"); plt.loglog(ranks, np.exp(intercept) * ranks**slope, label=f"fit slope={slope:.2f}")
    plt.title("Zipf plot of normalized corpus terms"); plt.xlabel("rank (log)"); plt.ylabel("frequency (log)"); plt.legend(); plt.tight_layout(); plt.savefig(FIGURES / "zipf_law.png", dpi=160); plt.close()
    plt.figure(figsize=(10, 7)); pos = nx.spring_layout(keyword_graph, seed=17, weight="weight")
    nx.draw_networkx_nodes(keyword_graph, pos, node_size=[500 + 13000 * degree[n] for n in keyword_graph], node_color=[next(i for i, c in enumerate(communities) if n in c) for n in keyword_graph], cmap="tab20", alpha=.85)
    nx.draw_networkx_edges(keyword_graph, pos, width=[.3 + d["weight"] / 4 for _, _, d in keyword_graph.edges(data=True)], alpha=.25)
    labels = {n: n for n in sorted(keyword_graph, key=lambda n: eigenvector[n], reverse=True)[:35]}; nx.draw_networkx_labels(keyword_graph, pos, labels, font_size=7)
    plt.title("Keyword co-occurrence graph (edges with ≥3 papers)"); plt.axis("off"); plt.tight_layout(); plt.savefig(FIGURES / "keyword_graph.png", dpi=180); plt.close()

    summary = {
        "query": QUERY, "records": len(records), "years": dict(sorted(years.items())), "tokens": len(all_tokens), "vocabulary": len(counts), "zipf_slope": float(slope),
        "keyword_graph": {"nodes": keyword_graph.number_of_nodes(), "edges": keyword_graph.number_of_edges(), "communities": len(communities), "modularity": modularity},
        "centrality_top10": centrality_rows[:10], "communities": community_rows, "publication_graph": {"nodes": publication_graph.number_of_nodes(), "edges": publication_graph.number_of_edges(), "density": nx.density(publication_graph)},
        "search_demo": {"query_id": query_record["id"], "query_title": query_record["title"], "results": search_results},
    }
    (DATA / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
