import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras import layers
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import f1_score

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# !pip install pymorphy3 -q

import pandas as pd

df = pd.read_csv("hf://datasets/Romjiik/Russian_bank_reviews/final_review_dataset_extended.csv")

morph=pymorphy3.MorphAnalyzer()
def preprocessing_text(str_row):
  s1 = re.sub(r'[^\w\s]+|[\d]+', r'',str_row).strip()
  s1 = s1.lower()
  word_arr = word_tokenize(s1)
  words=[]
  for i in word_arr:
        pv = morph.parse(i)
        words.append(pv[0].normal_form)
  sentence=' '.join(words)
  return sentence

df['preprocess_text'] = df['review'].apply(preprocessing_text)

def get_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['rating_value'].apply(get_sentiment)

russian_stopwords = stopwords.words("russian")

vectorizer_tfidf = TfidfVectorizer(max_features=500, min_df=20, max_df=0.7, stop_words=russian_stopwords)
text_tfidf = vectorizer_tfidf.fit_transform(df['preprocess_text'])
text_tfidf = pd.DataFrame(text_tfidf.toarray(),columns=vectorizer_tfidf.get_feature_names_out())
text_tfidf.head()

X = vectorizer_tfidf.fit_transform(df['preprocess_text'])

encoder = LabelEncoder()
y = encoder.fit_transform(df['sentiment'])
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.models.Sequential()
model.add(Dense(500, activation='relu', input_dim=X_train.shape[1]))
model.add(layers.Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=["accuracy"])

history=model.fit(X_train, y_train, epochs=5,batch_size=32)