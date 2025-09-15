# merge_similar_topics.py — improved fuzzy merge using token_sort_ratio at 80%
import json
from rapidfuzz import fuzz

INPUT_PATH = "grouped_articles_merged.json"
OUTPUT_PATH = "grouped_articles_merged.json"
SIMILARITY_THRESHOLD = 80


def merge_topics(topics):
    merged = []
    used = set()

    for i, current in enumerate(topics):
        if i in used:
            continue

        current_topic = current["topic"]
        current_articles = current["articles"]
        group = {
            "topic": current_topic,
            "articles": current_articles.copy()
        }

        for j, other in enumerate(topics[i + 1:], start=i + 1):
            if j in used:
                continue

            similarity = fuzz.token_sort_ratio(current_topic.lower(), other["topic"].lower())
            if similarity >= SIMILARITY_THRESHOLD:
                group["articles"].extend(other["articles"])
                used.add(j)

        merged.append(group)

    return merged


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        topic_groups = json.load(f)

    print(f"🔍 Loaded {len(topic_groups)} topic groups")
    merged = merge_topics(topic_groups)
    print(f"✅ Merged down to {len(merged)} unique topic groups")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"📦 Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
