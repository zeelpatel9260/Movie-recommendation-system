import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

print("Libraries loaded")


FILE_PATH = "/content/movies.csv"

def load_data():
    movies = pd.read_csv(FILE_PATH, on_bad_lines='skip')

    # Convert everything to string
    movies = movies.astype(str)

    # Normalize column names
    movies.columns = [c.strip().lower() for c in movies.columns]

    # Ensure required columns exist
    needed = ['title', 'genres', 'overview', 'keywords', 'cast', 'director']
    for col in needed:
        if col not in movies.columns:
            movies[col] = ""

    movies = movies.fillna("")
    return movies

movies = load_data()
print("\n✔ Dataset loaded successfully!")
print("Number of movies:", len(movies))

def combine_text(row):
    return (
        row['title'] + " " +
        row['genres'] + " " +
        row['keywords'] + " " +
        row['cast'] + " " +
        row['director'] + " " +
        row['overview']
    )

movies["combined"] = movies.apply(combine_text, axis=1)
print("\n✔ Combined text column created!")

# TF-IDF
def build_similarity(df):
    print("\n⏳ Building TF-IDF similarity matrix (may take a moment)…")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = tfidf.fit_transform(df["combined"])
    sim = cosine_similarity(matrix)
    print("✔ Similarity matrix created!")
    return sim

similarity = build_similarity(movies)

# Recommend Function
def recommend(movie_name, top_k=5):
    movie_name = movie_name.lower().strip()
    titles = movies["title"].str.lower().tolist()

    match = get_close_matches(movie_name, titles, n=1, cutoff=0.6)
    if not match:
        return None  # movie not found

    idx = titles.index(match[0])

    scores = list(enumerate(similarity[idx]))
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_k+1]

    results = []
    for (movie_idx, score) in scores_sorted:
        results.append({
            "title": movies.iloc[movie_idx]["title"],
            "genres": movies.iloc[movie_idx]["genres"],
            "score": round(float(score), 3)
        })
    return results

# Get recommendation from user
movie_name = input("\n🎬 Enter a movie name: ")

result = recommend(movie_name)

if result is None:
    print(f"\n❌ Movie '{movie_name}' not found in dataset.")

else:
    print(f"\n🎉 Recommendations similar to '{movie_name}':\n")
    for i, r in enumerate(result, start=1):
        print(f"{i}. {r['title']} — {r['genres']}")


# ==================================================
# STEP 7 — ADD MOVIE TO DATASET IF NOT FOUND
# ==================================================
if result is None:
    print("\n👇 Let's add this movie to the dataset.")

    title = input("Movie Title: ")
    genres = input("Genres: ")
    overview = input("Overview: ")
    keywords = input("Keywords: ")
    cast = input("Cast: ")
    director = input("Director: ")

    new_row = {
        "title": title,
        "genres": genres,
        "overview": overview,
        "keywords": keywords,
        "cast": cast,
        "director": director
    }

    movies = pd.concat([movies, pd.DataFrame([new_row])], ignore_index=True)
    movies.to_csv(FILE_PATH, index=False)

    print("\n✔ Movie added successfully to CSV!")

    # Rebuild combined text & similarity
    movies["combined"] = movies.apply(combine_text, axis=1)
    similarity = build_similarity(movies)

    print("\n🔁 Dataset updated!")


# ==================================================
# STEP 8 — TRY AGAIN
# ==================================================
movie_name = input("\n🔎 Enter the movie name again for recommendations: ")

result = recommend(movie_name)

if result is None:
    print("\n❌ Movie still not found.")
else:
    print(f"\n🎬 Recommendations for '{movie_name}':\n")
    for i, r in enumerate(result, start=1):
        print(f"{i}. {r['title']} — {r['genres']}")
