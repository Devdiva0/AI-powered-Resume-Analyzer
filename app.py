import streamlit as st
from utils import extract_text_from_pdf, extract_skills, predict_role, match_score, resume_score

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("🤖 AI Resume Analyzer (ATS System)")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")
job_desc = st.text_area("📝 Paste Job Description")

if uploaded_file:
    with st.spinner("Analyzing resume..."):
        text = extract_text_from_pdf(uploaded_file)

        # Skills
        skills = extract_skills(text)
        st.subheader("✅ Extracted Skills")
        st.write(skills)

        # Prediction
        role, confidence = predict_role(text)
        st.subheader("🎯 Predicted Job Role")
        st.success(f"{role} ({confidence}% confidence)")

        if job_desc:
            # Match score
            score = match_score(text, job_desc)
            st.subheader("📊 Match Score")
            st.progress(int(score))
            st.write(f"{score}% match")

            # Resume score
            final_score = resume_score(skills, score)
            st.subheader("🏆 Resume Score")
            st.success(f"{final_score}/100")

            # Suggestions
            st.subheader("💡 Suggestions")

            if score < 50:
                st.warning("Improve alignment with job description")
            if len(skills) < 5:
                st.warning("Add more technical skills")
            if confidence < 60:
                st.warning("Resume is unclear for a specific role")

            st.info("Tip: Customize resume for each job!")