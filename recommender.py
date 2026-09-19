import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, csv_path="movies_for_recommender.csv", max_features=20000):
        raw = pd.read_csv(csv_path)
        raw = raw.dropna(subset=["movie", "full_text_processed"])

        self.movies = raw.groupby("movie").agg(
            combined_text=("full_text_processed", lambda x: " ".join(x.astype(str))),
            avg_rating=("rating", "mean")
        ).reset_index()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies["combined_text"])

        self.title_to_index = pd.Series(self.movies.index, index=self.movies["movie"].str.lower())

    def search(self, query, limit=10):
        query = query.lower()
        matches = self.movies[self.movies["movie"].str.lower().str.contains(query, na=False)]
        return matches["movie"].head(limit).tolist()

    def recommend(self, title, top_n=3):
        title = title.lower()
        if title not in self.title_to_index:
            return []
        idx = self.title_to_index[title]

        query_vector = self.tfidf_matrix[idx]
        sims = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        scores = list(enumerate(sims))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        movie_indices = [i[0] for i in scores]
        return self.movies.iloc[movie_indices][["movie", "avg_rating"]].to_dict(orient="records")


recommender = MovieRecommender()