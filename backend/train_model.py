import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

df = df.rename(columns={"v1":"label","v2":"message"})
df = df[['label','message']]

# Convert labels
df['label'] = df['label'].map({'ham':0,'spam':1})

# Clean text
def clean_text(text):

    text = text.lower()
    text = re.sub(r'http\S+',' ',text)
    text = re.sub(r'\d+',' ',text)
    text = re.sub(r'[^\w\s]',' ',text)
    text = re.sub(r'\s+',' ',text)

    return text

df['message'] = df['message'].apply(clean_text)

# TF-IDF (OPTIMIZED)
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    max_features=10000,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df['message'])

y = df['label']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42
)

# Train BEST model
model = LinearSVC()

model.fit(X_train,y_train)

# Accuracy
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,y_pred))

# Save
pickle.dump(model,open("model.pkl","wb"))
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))
