import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

file_path = "amazon.csv"
df = pd.read_csv(file_path)
reviews = df[['review_content', 'rating']].dropna()

def label_sentiment(rating):
    try:
        rating = pd.to_numeric(rating, errors='coerce')  # Convert to numeric, set errors as NaN
        if pd.isna(rating):
            return 'neutral'  # Default to neutral if rating is invalid
        elif rating >= 4.0:
            return 'positive'
        elif rating == 3.0:
            return 'neutral'
        else:
            return 'negative'
    except Exception as e:
        return 'neutral'
reviews['sentiment'] = reviews['rating'].apply(label_sentiment)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)
reviews['cleaned_review'] = reviews['review_content'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(reviews['cleaned_review'])
y = reviews['sentiment']s
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

sentiment_counts = reviews['sentiment'].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'gray', 'red'])
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution of Amazon Reviews')
plt.show()
