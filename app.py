from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('ridge_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    review_text = ''

    if request.method == 'POST':
        review_text = request.form['review_text']
        vectorized = vectorizer.transform([review_text])
        raw_prediction = model.predict(vectorized)[0]

        clipped_prediction = max(1, min(10, raw_prediction))
        prediction = round(clipped_prediction, 1)

    return render_template('index.html', prediction=prediction, review_text=review_text)

if __name__ == '__main__':
    app.run(debug=True)