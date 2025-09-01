import sqlite3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DB_PATH = r"C:\Bachelorarbeit\Datenbank\news_cache.db"
MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.75

def get_existing_group_articles():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, article_text, stock, group_id FROM articles WHERE relevant = 1 AND group_id IS NOT NULL")
    data = cursor.fetchall()
    conn.close()
    return data

def get_new_articles_without_group():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, article_text, stock FROM articles WHERE relevant = 1 AND group_id IS NULL")
    data = cursor.fetchall()
    conn.close()
    return data

def get_next_group_number(stock_prefix, existing_group_ids):
    nums = [
        int(g.split("-")[1]) for g in existing_group_ids
        if g.startswith(stock_prefix + "-") and g.split("-")[1].isdigit()
    ]
    return max(nums, default=0) + 1

def assign_group_ids(new_articles, existing_articles, model):
    existing_texts = [a[1] for a in existing_articles]
    existing_embeddings = (
        list(model.encode(existing_texts, convert_to_tensor=False)) if existing_texts else []
    )

    existing_group_ids = [a[3] for a in existing_articles]
    new_group_assignments = []

    for new_id, new_text, new_stock in new_articles:
        if not new_text.strip():
            print(f"⚠️ Kein Text für Artikel {new_id} – übersprungen")
            continue

        new_embedding = model.encode([new_text], convert_to_tensor=False)[0]

        # Ähnlichkeit berechnen mit bestehenden Artikeln
        # Nur Gruppen mit passendem Aktien-Prefix zulassen
        stock_prefix = new_stock.strip()
        valid_indices = [
            idx for idx, gid in enumerate(existing_group_ids)
            if gid.startswith(f"{stock_prefix}-")
        ]

        best_score = 0
        best_idx = None

        if valid_indices:
            filtered_embeddings = [existing_embeddings[i] for i in valid_indices]
            similarities = cosine_similarity([new_embedding], filtered_embeddings)[0]
            best_local_idx = np.argmax(similarities)
            best_score = similarities[best_local_idx]
            best_idx = valid_indices[best_local_idx]

        if best_score >= SIMILARITY_THRESHOLD:
            assigned_group_id = existing_group_ids[best_idx]
            print(f"[GRUPPE GEFUNDEN] Artikel {new_id} → {assigned_group_id} (Score: {best_score:.2f})")

        else:
            prefix = new_stock.split(",")[0].strip() or "GENERIC"
            next_num = get_next_group_number(prefix, existing_group_ids)
            assigned_group_id = f"{prefix}-{next_num}"
            print(f"[NEUE GRUPPE] Artikel {new_id} → {assigned_group_id}")
            # Neue Gruppe als Referenz hinzufügen
            existing_embeddings.append(new_embedding)
            existing_group_ids.append(assigned_group_id)

        new_group_assignments.append((assigned_group_id, new_id))

    return new_group_assignments

def update_group_ids_in_db(assignments):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for group_id, article_id in assignments:
        cursor.execute("UPDATE articles SET group_id = ? WHERE id = ?", (group_id, article_id))
    conn.commit()
    conn.close()

def main():
    print("📥 Lade bestehende Gruppen-Artikel...")
    existing = get_existing_group_articles()
    print(f"✅ {len(existing)} bestehende Gruppen geladen")

    print("🔍 Lade neue Artikel ohne Gruppe...")
    new_articles = get_new_articles_without_group()
    print(f"📌 {len(new_articles)} neue Artikel zu prüfen")

    if not new_articles:
        print("✅ Keine neuen Artikel – alles gruppiert")
        return

    model = SentenceTransformer(MODEL_NAME)

    assignments = assign_group_ids(new_articles, existing, model)
    update_group_ids_in_db(assignments)
    print(f"✅ {len(assignments)} Artikel wurden gruppiert und aktualisiert.")

if __name__ == "__main__":
    main()
