import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Download stopwords if you haven't already
# nltk.download('stopwords')
from nltk.corpus import stopwords

# ===================== 1. Load Data =====================
# Load the CSV file (using latin-1 encoding to handle special characters)
df = pd.read_csv('spam.csv', encoding='latin-1')

print("Dataset shape:", df.shape)
print(df.head())

# Keep only the useful columns: 'v1' (label) and 'v2' (text)
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Map labels to numerical values: ham -> 0, spam -> 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("\nClass distribution:\n", df['label'].value_counts())

# ===================== 2. Text Preprocessing Function =====================
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and digits (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning to the text column
df['clean_text'] = df['text'].apply(clean_text)

# ===================== 3. Split Data into Train and Test Sets =====================
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print(f"\nTraining set size: {len(X_train)}, Test set size: {len(X_test)}")

# ===================== 4. Feature Extraction (TF-IDF) =====================

tfidf = TfidfVectorizer(
    max_features=5000,          # Keep only the top 5000 most important words
    stop_words='english',  # Use built-in English stopwords
    ngram_range=(1, 2),         # Use single words and two-word phrases
    min_df=2,                   # Ignore words that appear in fewer than 2 documents
    max_df=0.8                  # Ignore words that appear in more than 80% of documents
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"\nTraining feature matrix shape: {X_train_tfidf.shape}")

# ===================== 5. Model Training =====================
# Logistic Regression with balanced class weights (to handle the spam/ham imbalance)
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# ===================== 6. Model Evaluation =====================
y_pred = model.predict(X_test_tfidf)

print("\n======= Test Set Evaluation =======")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['ham', 'spam']))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ===================== 7. Predict New Messages =====================
def predict_message(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "spam" if pred == 1 else "ham"
    return label, prob

# Test with a few custom messages
new_messages = [
    "Congratulations! You've won a free iPhone. Call now!",
    "Hey, how are you? Let's meet tomorrow.",
    "URGENT! Your account has been suspended. Click here to reactivate."
]

print("\n======= Prediction on New Messages =======")
for msg in new_messages:
    label, prob = predict_message(msg)
    print(f"Message: {msg}")
    print(f"Prediction: {label} (Probability: ham={prob[0]:.3f}, spam={prob[1]:.3f})\n")