from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import re
import pickle
import joblib
from keras.models import load_model
from datetime import datetime
import json
from collections import defaultdict

app = Flask(__name__)

reviews_storage = []
sentiment_stats = {'positive': 0, 'neutral': 0, 'negative': 0}
bank_stats = defaultdict(int)

try:
    model = load_model('model/model.keras')
    with open('model/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    encoder = joblib.load('model/label_encoder.joblib')
except Exception as e:
    print(e)
    model = None

def preprocess_text(text):
    text = re.sub(r'[^\w\s]+|[\d]+', r'', text).strip()
    text = text.lower()
    words = text.split()
    return ' '.join(words)

def predict_sentiment(text):
    if model is None:
        return None, 0.0
    
    processed = preprocess_text(text)
    X = vectorizer.transform([processed])
    prediction = model.predict(X.toarray())
    class_idx = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    sentiment = encoder.inverse_transform([class_idx])[0]
    
    return sentiment, confidence

def add_review(bank, text, sentiment, confidence):
    review = {
        'id': len(reviews_storage) + 1,
        'bank': bank,
        'text': text,
        'sentiment': sentiment,
        'confidence': confidence,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'sentiment_ru': {
            'positive': 'Позитивный',
            'neutral': 'Нейтральный',
            'negative': 'Негативный'
        }[sentiment]
    }
    reviews_storage.append(review)
    
    sentiment_stats[sentiment] += 1
    bank_stats[bank] += 1
    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

# Обработка отзыва
@app.route('/submit_review', methods=['POST'])
def submit_review():
    bank = request.form.get('bank', 'Не указан')
    text = request.form.get('text', '')
    
    if not text:
        return redirect('/')
    
    sentiment, confidence = predict_sentiment(text)
    
    if sentiment:
        add_review(bank, text, sentiment, confidence)
        
        return render_template('index.html', 
                             result=True,
                             sentiment=sentiment,
                             confidence=f"{confidence:.1%}",
                             text=text[:200])
    else:
        return render_template('index.html', error="Ошибка анализа")

# для получения статистики
@app.route('/api/stats')
def get_stats():
    total = sum(sentiment_stats.values())
    
    return jsonify({
        'total_reviews': total,
        'sentiment_distribution': sentiment_stats,
        'bank_distribution': dict(bank_stats),
        'avg_confidence': np.mean([r['confidence'] for r in reviews_storage]) if reviews_storage else 0
    })

# для получения всех отзывов
@app.route('/api/reviews')
def get_reviews():
    return jsonify(reviews_storage[-20:])  # Последние 20 отзывов

# REST API для предсказания
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Нет текста'}), 400
    
    text = data['text']
    sentiment, confidence = predict_sentiment(text)
    
    if not sentiment:
        return jsonify({'error': 'Модель не загружена'}), 500
    
    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence,
        'label_ru': 'позитивный' if sentiment == 'positive' else 
                   'нейтральный' if sentiment == 'neutral' else 'негативный'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)