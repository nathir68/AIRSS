import streamlit as st
import PyPDF2
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io


import io

def extract_text_from_pdf(file):
    try:
        
        file_bytes = file.read()
        if not file_bytes:
            return "" 
            
        pdf_stream = io.BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_stream)
        
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
    except Exception as e:
        
        st.error(f"Could not read {file.name}: The file might be corrupted.")
        return ""

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
    return " ".join(text.split())


st.set_page_config(page_title="AI Resume Ranker", layout="wide")

st.title("🤖 Intelligent Resume Screening System")
st.markdown("Measuring match accuracy using **TF-IDF Vectorization** and **Cosine Similarity**.")


st.sidebar.header("Step 1: Job Description")
jd_text = st.sidebar.text_area("Paste the Job Description here:", height=300)


st.header("Step 2: Upload Resumes")
uploaded_files = st.file_uploader("Upload candidate resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("Rank Resumes"):
    if jd_text and uploaded_files:
        with st.spinner("Calculating Match Accuracy..."):
            resumes_content = []
            filenames = []
            
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                resumes_content.append(clean_text(text))
                filenames.append(file.name)
            
            
            cleaned_jd = clean_text(jd_text)
            corpus = [cleaned_jd] + resumes_content
            
            
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            
            scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            
            results = pd.DataFrame({
                "Candidate": filenames,
                "Match Accuracy (%)": [round(score * 100, 2) for score in scores]
            }).sort_values(by="Match Accuracy (%)", ascending=False)
            
            
            top_score = results.iloc[0]["Match Accuracy (%)"]
            avg_score = results["Match Accuracy (%)"].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Top Match Accuracy", f"{top_score}%")
            col2.metric("Average Pool Match", f"{round(avg_score, 2)}%")
            col3.metric("Model Confidence", "High" if top_score > 50 else "Moderate")

            st.success("Analysis Complete!")
            st.table(results)
            st.balloons()
    else:
        st.warning("Please provide both a Job Description and at least one Resume.")