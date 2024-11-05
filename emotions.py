import joblib
from tensorflow.keras.models import load_model
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk

model = load_model('models/emotion_model.h5')
label_encoder = joblib.load('models/label_encoder.joblib')
cv = joblib.load('models/count_vectorizer.joblib')

porter = PorterStemmer()
nltk.download("stopwords")

def preprocess(line):
    review = re.sub("[^a-zA-z]", " ", line)
    review = review.lower()
    review = review.split()
    review = [porter.stem(word) for word in review if not word in stopwords.words("english")]
    return " ".join(review)

def predictEmotion(text):
    text = preprocess(text)
    array = cv.transform([text]).toarray()
    pred = model.predict(array)
    emotion = label_encoder.inverse_transform(range(pred.shape[1]))
    emotion_percentages = {emotion[i]: float(round(pred[0][i] * 100, 2)) for i in range(len(emotion))}
    return emotion_percentages