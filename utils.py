import PyPDF2
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Extract text
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Extract skills
def extract_skills(text):
    df = pd.read_csv("skills.csv")
    skills_list = df['skills'].dropna().tolist()
    
    found = []
    for skill in skills_list:
        if skill.lower() in text.lower():
            found.append(skill)
    
    return list(set(found))

# Predict role
def predict_role(text):
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max() * 100
    return pred, round(prob, 2)

# Match score
def match_score(resume, jd):
    vec = tfidf.transform([resume, jd])
    score = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    return round(score * 100, 2)

# Resume score
def resume_score(skills, match):
    score = min(len(skills) * 5, 40) + (match * 0.6)
    return round(score, 2)