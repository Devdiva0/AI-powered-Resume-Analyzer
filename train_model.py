import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("UpdatedResumeDataSet.csv")

# Clean column names if needed
df.columns = df.columns.str.strip()

X = df['Resume']
y = df['Category']

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = tfidf.fit_transform(X)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

print("✅ Model trained successfully!")