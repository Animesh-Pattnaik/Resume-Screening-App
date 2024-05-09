import streamlit as st
import PyPDF2
import docx2txt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

def extract_text_from_pdf(file_path):
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        page_obj = pdf_reader.pages[page]
        text += page_obj.extract_text()
    pdf_file.close()
    return text

def extract_text_from_docx(file_path):
    text = docx2txt.process(file_path)
    return text

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in word_tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens

def calculate_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity_score = cosine_similarity(vectors)[0][1]
    return similarity_score

def generate_wordcloud(text, title):
    word_cloud = WordCloud(collocations=False, background_color='white').generate(text)
    st.subheader(title)
    st.image(word_cloud.to_array(), use_column_width=True)

def main():
    st.set_page_config(page_title="Resume Screener", layout="wide")
    st.title("Resume Screening App")

    # Sidebar with general description of the app
    st.sidebar.title("About the App")
    st.sidebar.markdown(
        """
        <div style='border: 2px solid #ccc; padding: 10px; height: 200px; display: flex; justify-content: center; padding-top:30px'>
        Welcome to the Resume Screening App. Upload a Job Description and a Resume to see their similarity and WordCloud and screen through Resume.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Create placeholders for file paths
    jd_file_path = None
    resume_file_path = None

    st.subheader("Upload Documents")

    with st.expander("Upload Job Description Document (PDF or DOCX)"):
        jd_file = st.file_uploader("Upload Job Description", type=['pdf', 'docx'], key="jd_file")
        if jd_file is not None:
            jd_file_path = save_uploaded_file(jd_file)
            jd_text = extract_text_from_docx(jd_file_path) if jd_file.name.endswith('.docx') else extract_text_from_pdf(jd_file_path)
            st.success("Job Description uploaded successfully!")

    with st.expander("Upload Resume Document (PDF or DOCX)"):
        resume_file = st.file_uploader("Upload Resume", type=['pdf', 'docx'], key="resume_file")
        if resume_file is not None:
            resume_file_path = save_uploaded_file(resume_file)
            resume_text = extract_text_from_docx(resume_file_path) if resume_file.name.endswith('.docx') else extract_text_from_pdf(resume_file_path)
            st.success("Resume uploaded successfully!")

    if jd_file_path is not None and resume_file_path is not None:
        similarity_score = calculate_similarity(resume_text, jd_text)
        st.info(f"The similarity score between your resume and the job description is: **{similarity_score:.3f}**")
        st.text("")
        col1, col2 = st.columns(2)
        with col1:
            generate_wordcloud(jd_text, "Word Cloud of Job Description")
        with col2:
            generate_wordcloud(resume_text, "Word Cloud of Resume")

# Function to save the uploaded file to a temporary location and return its path
def save_uploaded_file(uploaded_file):
    import tempfile

    _, temp_file_path = tempfile.mkstemp()
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.getvalue())

    return temp_file_path

if __name__ == "__main__":
    main()
