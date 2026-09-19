# Rating Fortune

A Flask web app that predicts an IMDb rating from a movie review and recommends similar movies.

## Features

- **Rating prediction:** TF-IDF + Ridge Regression and a sentiment model, trained on IMDb reviews.
- **Movie recommendations:** Content-based filtering using TF-IDF and cosine similarity on the reviews of each movie.

## Run Locally

```bash
git clone https://github.com/birce99/imdb_rating.git
cd imdb_rating
pip install flask pandas numpy scikit-learn
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Required Files

These files are too large for GitHub and are not in the repo. Place them next to `app.py`:

- `movies_for_recommender.csv`: [download link]
- Trained `.pkl` model files: [download link]

## Author

Birce Duasu Haşcelik
