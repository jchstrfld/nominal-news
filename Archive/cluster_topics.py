# cluster_topics.py (Now Clustering All Articles for More Overlap)
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict

credibility_scores = {
    "Associated Press": 1.0,
    "AP": 1.0,
    "Reuters": 1.0,
    "AFP": 1.0,
    "NPR": 1.0,
    "BBC News": 0.9,
    "Deutsche Welle": 0.9,
    "The Wall Street Journal": 0.9,
    "The New York Times": 0.9,
    "Bloomberg": 0.9,
    "Al Jazeera": 0.9,
    "NHK WORLD-JAPAN": 0.9,
    "ABC News (Australia)": 0.9,
    "CBC": 0.9,
    "The Guardian": 0.8,
    "CNN": 0.8,
    "Politico": 0.8,
    "The Hill": 0.8,
    "Yahoo News": 0.8,
    "Fox News": 0.7,
    "MSNBC": 0.7,
    "The Telegraph": 0.7,
    "India Today": 0.7,
    "Daily Wire": 0.6,
    "The Blaze": 0.6,
    "Jacobin": 0.6,
    "Democracy Now": 0.6,
    "Breitbart": 0.5,
    "Epoch Times": 0.5,
    "OANN": 0.4,
    "The Gateway Pundit": 0.3,
    "unknown": 0.0
}

def load_articles(path="articles_with_bias.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def embed_articles(articles):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [
        (article.get("title") or "") + " " + (article.get("description") or "")
        for article in articles
    ]
    embeddings = model.encode(texts)
    return embeddings

def cluster_articles(embeddings, eps=0.55, min_samples=3):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clustering.fit_predict(embeddings)
    return labels

def group_articles_by_cluster(articles, labels):
    clusters = defaultdict(list)
    for article, label in zip(articles, labels):
        if label == -1:
            continue  # skip outliers
        clusters[label].append(article)
    return clusters

def save_clusters(clusters, output_path="clustered_articles.json"):
    output = []
    for cluster_id, articles in clusters.items():
        output.append({
            "topic_id": int(cluster_id),
            "articles": articles
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Saved {len(output)} clusters to {output_path}")

def main():
    all_articles = load_articles()
    embeddings = embed_articles(all_articles)
    labels = cluster_articles(embeddings, eps=0.55, min_samples=3)
    clusters = group_articles_by_cluster(all_articles, labels)
    save_clusters(clusters)

if __name__ == "__main__":
    main()
