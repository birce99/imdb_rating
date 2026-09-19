from flask import Flask, render_template, request, jsonify
from recommender import recommender
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from scipy.sparse import hstack
import nltk

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

app = Flask(__name__)


# Ana rating modeli (TF-IDF 20k + sentiment_proba feature ile eğitildi, R2=0.617)
model = joblib.load('ridge_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Kısa yorumlar icin ayri egitilmis sentiment siniflandiricisi;
# artik duzeltme olarak degil, ana modele giden bir feature olarak kullaniliyor
sentiment_clf = joblib.load('sentiment_model.pkl')
sentiment_vectorizer = joblib.load('sentiment_vectorizer.pkl')

lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'


def preprocess_text(text):
    text = text.lower()

    # Negasyon isleme: "not good" -> "not_good"
    text = re.sub(r"\b(not|no|never|n't)\s+(\w+)", r"\1_\2", text)

    # Basit temizlik
    text = re.sub(r"[^a-z0-9_\s]", " ", text)

    # POS-aware lemmatization
    words = text.split()
    tagged = pos_tag(words)
    words = [lemmatizer.lemmatize(w, get_wordnet_pos(tag)) for w, tag in tagged]

    return ' '.join(words)


def get_sentiment_proba(processed_text):
    vec = sentiment_vectorizer.transform([processed_text])
    return sentiment_clf.predict_proba(vec)[0][1]  # pozitif olasiligi


def predict_rating(review_text, word_limit=10):
    processed = preprocess_text(review_text)

    sentiment_proba = get_sentiment_proba(processed)

    tfidf_vec = vectorizer.transform([processed])
    combined = hstack([tfidf_vec, [[sentiment_proba]]])

    raw_prediction = model.predict(combined)[0]
    clipped_prediction = max(1, min(10, raw_prediction))

    # Kisa ve net duygulu yorumlar icin ek guvenlik agi:
    # ayni sentiment_proba'yi kullanarak asiri net durumlarda tahmini sinirla
    word_count = len(processed.split())
    if word_count <= word_limit:
        print(sentiment_proba)
        print(sentiment_proba > 0.6)
        if sentiment_proba < 0.4:
            clipped_prediction = max(clipped_prediction, 3.0)
        elif sentiment_proba > 0.6:
            clipped_prediction = min(clipped_prediction, 8.0)

    return clipped_prediction


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    review_text = ''

    if request.method == 'POST':
        review_text = request.form['review_text']
        prediction = round(predict_rating(review_text), 1)
        print(f"prediction ",prediction)

    return render_template('index.html', prediction=prediction, review_text=review_text)

@app.route('/search')
def search_movies():
    query = request.args.get('q', '')
    results = recommender.search(query)
    return jsonify({'results': results})

@app.route('/recommend')
def recommend_movies():
    title = request.args.get('title', '')
    results = recommender.recommend(title)
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)