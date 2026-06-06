import os
import sys
import re
import ast
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from src.parser import parse_resume, extract_skills_from_text, parse_education_level, parse_required_experience

# Set Streamlit Page Config
st.set_page_config(
    page_title="RecruitAI Pro | Smart Recruitment Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

# ----------------- Theme CSS (Modern Clean Design) -----------------
def get_theme_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@500;600;700&display=swap');
        
        :root {
            --bg-main: #f8fafc;
            --bg-card: #ffffff;
            --border: #e2e8f0;
            --text-high: #1e293b;
            --text-mid: #64748b;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }
        
        .stApp {
            background-color: var(--bg-main) !important;
            color: var(--text-high) !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Poppins', sans-serif !important;
            color: var(--text-high) !important;
            font-weight: 600 !important;
        }
        
        /* Header Styling */
        .header-container {
            background-color: white;
            border-bottom: 1px solid var(--border);
            padding: 15px 30px;
            margin-bottom: 25px;
            box-shadow: var(--shadow);
        }
        
        .logo {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 1.75rem;
            color: var(--accent);
        }
        
        .tagline {
            font-size: 0.9rem;
            color: var(--text-mid);
            margin-top: 5px;
        }
        
        /* Card Styling */
        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            box-shadow: var(--shadow);
            margin-bottom: 20px;
        }
        
        .card-header {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1.1rem;
            color: var(--text-high);
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }
        
        /* Metric Cards */
        .metric-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px;
            box-shadow: var(--shadow);
            text-align: center;
            transition: all 0.2s ease;
        }
        
        .metric-card:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }
        
        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 0.8rem;
            color: var(--text-mid);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Button Styling */
        .stButton > button {
            background-color: var(--accent) !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton > button:hover {
            background-color: var(--accent-hover) !important;
            box-shadow: var(--shadow-md) !important;
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-mid);
            padding: 8px 16px;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--accent) !important;
            color: white !important;
            border: none !important;
        }
        
        /* Skill Tags */
        .skill-tag {
            display: inline-block;
            background-color: #eff6ff;
            color: var(--accent);
            border: 1px solid #bfdbfe;
            border-radius: 4px;
            padding: 3px 8px;
            font-size: 0.8rem;
            margin: 2px;
        }
        
        .matched-skill {
            background-color: #ecfdf5;
            color: var(--success);
            border-color: #a7f3d0;
        }
        
        .missing-skill {
            background-color: #fef2f2;
            color: var(--error);
            border-color: #fecaca;
        }
        
        /* Progress Bar */
        .progress-container {
            background-color: #e2e8f0;
            border-radius: 4px;
            height: 8px;
            margin-top: 5px;
        }
        
        .progress-bar {
            height: 8px;
            border-radius: 4px;
            background-color: var(--accent);
        }
        
        /* Hide default Streamlit elements */
        [data-testid="collapsedControl"] {
            display: none;
        }
        
        header {
            visibility: hidden;
        }
    </style>
    """

# ----------------- Cache Assets Loader -----------------
@st.cache_resource(show_spinner="Loading recruitment models...")
def load_assets():
    try:
        with open(os.path.join(BASE_DIR, '../models/skills_vocab.pkl'), 'rb') as f:
            skills_vocab = pickle.load(f)
        with open(os.path.join(BASE_DIR, '../models/tfidf_vectorizer.pkl'), 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open(os.path.join(BASE_DIR, '../models/match_model.pkl'), 'rb') as f:
            match_model = pickle.load(f)
        with open(os.path.join(BASE_DIR, '../models/metrics.pkl'), 'rb') as f:
            model_metrics = pickle.load(f)
        return skills_vocab, tfidf_vectorizer, match_model, model_metrics
    except Exception as e:
        st.error(f"Failed to load trained model assets. Ensure `model.py` was executed. Details: {e}")
        return None, None, None, None

@st.cache_data(show_spinner=False)
def load_bert_embeddings():
    path = os.path.join(BASE_DIR, '../models/bert_embeddings.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}

@st.cache_resource(show_spinner="Loading BERT model...")
def load_bert_model():
    import torch
    torch.set_num_threads(4)
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model.eval()
    return tokenizer, model

# Load assets
skills_vocab, tfidf_vectorizer, match_model, model_metrics = load_assets()
bert_embeddings = load_bert_embeddings()

# Demo Job Descriptions
def load_demo_jobs():
    return {
        "Senior Software Engineer": {
            "title": "Senior Software Engineer",
            "text": "Senior Software Engineer\nWe are looking for a Senior Software Engineer with strong expertise in Java, Spring Boot, MySQL, and cloud services (AWS or Azure). The ideal candidate must have at least 5 years of software development experience, design robust microservices architecture, and participate in unit testing, debugging, and continuous integration. Required skills: Java, Spring Boot, SQL, AWS, Docker, Microservices, Git, JUnit.",
            "skills": "Java, Spring Boot, SQL, AWS, Docker, Microservices, Git, JUnit",
            "experience": 5,
            "education": "Bachelor's Degree in Computer Science"
        },
        "Machine Learning Engineer": {
            "title": "Machine Learning Engineer",
            "text": "Machine Learning Engineer / Data Scientist\nSeeking an ML Engineer with 3+ years of experience training predictive models. Candidate must be skilled in Python, Scikit-learn, TensorFlow, PyTorch, Pandas, and SQL. Responsibilities include building data pipelines, statistical data analysis, model deployment, and NLP keyword extraction. Required skills: Python, Machine Learning, TensorFlow, PyTorch, Scikit-learn, Pandas, NLP, SQL.",
            "skills": "Python, Machine Learning, TensorFlow, PyTorch, Scikit-learn, Pandas, NLP, SQL",
            "experience": 3,
            "education": "Master's Degree in Computer Science or Statistics"
        },
        "Senior Business Analyst": {
            "title": "Senior Business Analyst (BSA)",
            "text": "Senior Business Systems Analyst\nWe need a Senior Business Analyst with a minimum of 4 years of experience gathering requirements, writing detailed user stories, creating process flow charts in Visio, and coordinating between engineering teams and stakeholders. Must have strong understanding of Agile / Scrum methodologies. Required skills: Requirement Gathering, Agile, Scrum, Jira, MS Visio, Business Process Mapping, User Stories, UAT Testing.",
            "skills": "Requirement Gathering, Agile, Scrum, Jira, MS Visio, Business Process Mapping, User Stories, UAT Testing",
            "experience": 4,
            "education": "Bachelor's Degree"
        },
        "Technical Project Manager / Scrum Master": {
            "title": "Technical Project Manager (PM / Scrum Master)",
            "text": "Technical Program Manager / Scrum Master\nWe are hiring a Scrum Master and Project Manager to lead agile sprint teams. Requirements: 6+ years experience, PMP or Scrum Master (CSM) certification, expertise in Jira, team leadership, risk management, and software development lifecycles. Required skills: Project Management, Agile, Scrum, Scrum Master, Jira, PMP, Risk Management, Team Leadership, SDLC.",
            "skills": "Project Management, Agile, Scrum, Scrum Master, Jira, PMP, Risk Management, Team Leadership, SDLC",
            "experience": 6,
            "education": "Bachelor's or Master's Degree"
        }
    }

demo_jobs = load_demo_jobs()

# BERT Embedding Helper
def get_bert_embedding(text, tokenizer, model):
    import torch
    
    if text and len(text) > 1500:
        text = text[:1500]
    
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings[0].numpy()

# Custom Candidate Redaction for Blind Screening
def redact_profile(profile, candidate_idx):
    return {
        'filename': profile['filename'],
        'filepath': profile['filepath'],
        'name': f"Candidate #{candidate_idx}",
        'email': "[REDACTED]@blind-screen.com",
        'phone': "[REDACTED]",
        'location': "Hidden (Blind Screening)",
        'education_degree': profile['education_degree'],
        'education_level': profile['education_level'],
        'experience_years': profile['experience_years'],
        'skills': profile['skills'],
        'text': profile['text']
    }

# Apply theme
st.markdown(get_theme_css(), unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="logo">RecruitAI Pro</div>
            <div class="tagline">Intelligent Recruitment Platform for Modern Hiring</div>
        </div>
        <div style="display: flex; gap: 10px;">
            <button style="background-color: white; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 16px; color: #64748b; font-weight: 500; cursor: pointer;">Documentation</button>
            <button style="background-color: #3b82f6; border: none; border-radius: 6px; padding: 8px 16px; color: white; font-weight: 500; cursor: pointer;">About</button>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'parsed_uploads' not in st.session_state:
    st.session_state.parsed_uploads = {}

# Check model assets
if not match_model:
    st.warning("⚠️ Application is running in fallback mode. Please run `python src/model.py` to compile model assets.")
    st.stop()

# Sidebar for navigation and configuration
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Select a Page", ["Dashboard", "Resume Analytics", "Candidate Ranking", "Explainability", "Settings"])
    
    st.markdown("### Quick Settings")
    blind_screening = st.checkbox("Blind Screening", value=True, help="Hide candidate identity to reduce bias")
    min_match_score = st.slider("Minimum Match Score", min_value=0, max_value=100, value=50)
    min_experience = st.slider("Minimum Experience (Years)", min_value=0, max_value=20, value=0)
    match_engine = st.selectbox("Matching Engine", ["High-End Semantic Match (Recommended)", "TF-IDF Cosine Similarity"])

# Main content area
if page == "Dashboard":
    st.markdown("### Recruitment Dashboard")
    
    # Job Description and Resume Upload
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card"><div class="card-header">Job Description</div>', unsafe_allow_html=True)
        jd_preset = st.selectbox("Select a Job Opening", ["Custom Input"] + list(demo_jobs.keys()))
        
        if jd_preset != "Custom Input":
            preset_info = demo_jobs[jd_preset]
            jd_title = preset_info['title']
            jd_text_val = preset_info['text']
            req_exp_val = preset_info['experience']
        else:
            jd_title = "Custom Job Opening"
            jd_text_val = ""
            req_exp_val = 0
            
        jd_text = st.text_area("Paste Job Description", value=jd_text_val, height=200)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card"><div class="card-header">Upload Resumes</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload Candidate Resumes", 
            type=["docx", "pdf", "txt"], 
            accept_multiple_files=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Parse uploaded files
    raw_candidates = []
    if uploaded_files:
        for f in uploaded_files:
            cache_key = f"{f.name}_{f.size}"
            if cache_key in st.session_state.parsed_uploads:
                raw_candidates.append(st.session_state.parsed_uploads[cache_key])
            else:
                temp_dir = os.path.join(BASE_DIR, '../temp_uploads')
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f.name)
                with open(temp_path, 'wb') as temp_f:
                    temp_f.write(f.read())
                
                try:
                    ext = f.name.lower()
                    if ext.endswith(('.docx', '.pdf', '.txt')):
                        profile = parse_resume(temp_path, skills_vocab)
                        st.session_state.parsed_uploads[cache_key] = profile
                        raw_candidates.append(profile)
                except Exception as e:
                    st.error(f"Error parsing file {f.name}: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    # Apply Blind Screening if selected
    candidates = []
    for idx, profile in enumerate(raw_candidates):
        if blind_screening:
            candidates.append(redact_profile(profile, idx+1))
        else:
            candidates.append(profile)
    
    # Metrics Overview
    if candidates:
        st.markdown("### Overview Metrics")
        metrics_cols = st.columns(4)
        
        with metrics_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(candidates)}</div>
                <div class="metric-label">Candidates</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(jd_text.split())}</div>
                <div class="metric-label">JD Words</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(jd_text.split('.'))}</div>
                <div class="metric-label">JD Sentences</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_cols[3]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(extract_skills_from_text(jd_text, skills_vocab))}</div>
                <div class="metric-label">JD Skills</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Process button
        if st.button("Process Resumes", type="primary"):
            with st.spinner("Analyzing candidates..."):
                # Extract JD values
                jd_skills = extract_skills_from_text(jd_text, skills_vocab)
                jd_min_exp = parse_required_experience(jd_text) if jd_preset == "Custom Input" else req_exp_val
                jd_edu_level = parse_education_level(jd_text)
                
                # Initialize BERT model if chosen
                if match_engine.startswith("High-End Semantic"):
                    tokenizer, model = load_bert_model()
                
                # Calculate Match Scores
                scored_candidates = []
                for profile in candidates:
                    try:
                        # Calculate common structural features
                        common_skills = list(set(profile['skills']).intersection(set(jd_skills)))
                        overlap_count = len(common_skills)
                        overlap_ratio = overlap_count / len(jd_skills) if len(jd_skills) > 0 else 0.0
                        
                        c_exp = profile['experience_years']
                        exp_diff = c_exp - jd_min_exp
                        
                        c_edu = profile['education_level']
                        edu_diff = c_edu - jd_edu_level

                        # Compute match score based on selected engine
                        if match_engine.startswith("High-End Semantic"):
                            # Encode JD
                            j_combined_text = (jd_title + " " + jd_text)
                            j_emb = get_bert_embedding(j_combined_text, tokenizer, model)
                            
                            # Retrieve cached candidate embedding or encode on-the-fly
                            c_emb = bert_embeddings.get(profile['filename'])
                            if c_emb is None:
                                c_combined_text = (
                                    f"Name: {profile['name']}\n"
                                    f"Education: {profile['education_degree']}\n"
                                    f"Experience: {profile['experience_years']} years\n"
                                    f"Skills: {', '.join(profile['skills'])}\n"
                                    f"Content: {profile['text']}"
                                )
                                c_emb = get_bert_embedding(c_combined_text, tokenizer, model)
                                
                            # Compute Cosine similarity
                            norm_c = np.linalg.norm(c_emb)
                            norm_j = np.linalg.norm(j_emb)
                            sim = np.dot(c_emb, j_emb) / (norm_c * norm_j) if (norm_c > 0 and norm_j > 0) else 0.0
                            
                            # Scale similarity [0.2, 0.8] -> [0, 100]%
                            final_score = int(np.clip((sim - 0.2) / 0.6 * 100, 0, 100))
                        else:
                            # TF-IDF Cosine Similarity
                            c_combined_text = (profile['name'] + " " + " ".join(profile['skills']) + " " + profile['text'])
                            j_combined_text = (jd_title + " " + jd_text)
                            
                            c_tfidf = tfidf_vectorizer.transform([c_combined_text])
                            j_tfidf = tfidf_vectorizer.transform([j_combined_text])
                            sim = cosine_similarity(c_tfidf, j_tfidf)[0][0]
                            
                            # Scale similarity [0.0, 1.0] -> [0, 100]%
                            final_score = int(np.clip(sim * 100, 0, 100))
                        
                        scored_candidates.append({
                            'profile': profile,
                            'score': final_score,
                            'common_skills': common_skills,
                            'missing_skills': list(set(jd_skills) - set(profile['skills'])),
                            'similarity': sim if 'sim' in locals() else 0.0
                        })
                    except Exception as e:
                        st.error(f"Error scoring {profile['name']}: {e}")
                
                # Store in session state
                st.session_state.scored_candidates = scored_candidates
                st.success(f"Successfully processed {len(scored_candidates)} candidates!")
    else:
        st.info("ℹ️ Please upload resumes and provide a job description to begin analysis.")

elif page == "Resume Analytics":
    st.markdown("### Resume Analytics")
    
    if 'scored_candidates' in st.session_state and st.session_state.scored_candidates:
        candidates = [c['profile'] for c in st.session_state.scored_candidates]
        
        # Create tabs for different analytics
        tab1, tab2, tab3 = st.tabs(["Skill Distribution", "Experience Analysis", "Education Overview"])
        
        with tab1:
            # Skill Word Cloud
            all_skills = []
            for candidate in candidates:
                all_skills.extend(candidate['skills'])
            
            skill_counts = pd.DataFrame(pd.Series(all_skills).value_counts()).reset_index()
            skill_counts.columns = ['Skill', 'Count']
            skill_counts = skill_counts.head(20)  # Top 20 skills
            
            fig = px.bar(
                skill_counts, 
                x='Count', 
                y='Skill',
                orientation='h',
                color='Count',
                color_continuous_scale='Blues',
                title='Top Skills Distribution'
            )
            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Experience Distribution
            exp_data = pd.DataFrame([{
                'Candidate': c['name'],
                'Years of Experience': c['experience_years']
            } for c in candidates])
            
            fig = px.histogram(
                exp_data,
                x='Years of Experience',
                nbins=10,
                title='Experience Distribution',
                color_discrete_sequence=['#3b82f6']
            )
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Education Level Distribution
            edu_data = pd.DataFrame([{
                'Education Level': c['education_degree'],
                'Count': 1
            } for c in candidates])
            
            edu_counts = edu_data.groupby('Education Level').count().reset_index()
            
            fig = px.pie(
                edu_counts,
                values='Count',
                names='Education Level',
                title='Education Level Distribution',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Please process resumes first to view analytics.")

elif page == "Candidate Ranking":
    st.markdown("### Candidate Ranking")
    
    if 'scored_candidates' in st.session_state and st.session_state.scored_candidates:
        scored_candidates = st.session_state.scored_candidates
        
        # Filter candidates based on settings
        filtered_candidates = [
            c for c in scored_candidates 
            if c['score'] >= min_match_score and c['profile']['experience_years'] >= min_experience
        ]
        
        # Sort by score
        filtered_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Display top 10 candidates
        st.markdown(f"#### Top {min(10, len(filtered_candidates))} Candidates")
        
        for i, candidate in enumerate(filtered_candidates[:10]):
            profile = candidate['profile']
            score = candidate['score']
            
            # Determine score color
            if score >= 80:
                score_color = "#10b981"  # Green
            elif score >= 60:
                score_color = "#3b82f6"  # Blue
            elif score >= 40:
                score_color = "#f59e0b"  # Yellow
            else:
                score_color = "#ef4444"  # Red
            
            with st.expander(f"{i+1}. {profile['name']} - {score}% Match"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Experience:** {profile['experience_years']} years")
                    st.markdown(f"**Education:** {profile['education_degree']}")
                    
                    st.markdown("**Skills:**")
                    for skill in profile['skills']:
                        if skill in candidate['common_skills']:
                            st.markdown(f"<span class='skill-tag matched-skill'>{skill}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span class='skill-tag'>{skill}</span>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px; background-color: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
                        <div style="font-size: 2.5rem; font-weight: 700; color: {score_color};">{score}%</div>
                        <div style="font-size: 0.85rem; color: #64748b; margin-top: 5px;">Match Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if candidate['missing_skills']:
                        st.markdown("**Missing Skills:**")
                        for skill in candidate['missing_skills']:
                            st.markdown(f"<span class='skill-tag missing-skill'>{skill}</span>", unsafe_allow_html=True)
    else:
        st.info("ℹ️ Please process resumes first to view rankings.")

elif page == "Explainability":
    st.markdown("### Model Explainability")
    
    if 'scored_candidates' in st.session_state and st.session_state.scored_candidates:
        # Select a candidate to explain
        candidate_names = [c['profile']['name'] for c in st.session_state.scored_candidates]
        selected_name = st.selectbox("Select a Candidate to Explain", candidate_names)
        
        # Find the selected candidate
        selected_candidate = None
        for candidate in st.session_state.scored_candidates:
            if candidate['profile']['name'] == selected_name:
                selected_candidate = candidate
                break
        
        if selected_candidate:
            profile = selected_candidate['profile']
            score = selected_candidate['score']
            
            # Display explanation
            st.markdown(f"#### Why was {profile['name']} selected?")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Key Strengths**")
                
                # Common skills
                st.markdown(f"**Matched Skills ({len(selected_candidate['common_skills'])}/{len(extract_skills_from_text(jd_text, skills_vocab))}):**")
                for skill in selected_candidate['common_skills']:
                    st.markdown(f"<span class='skill-tag matched-skill'>{skill}</span>", unsafe_allow_html=True)
                
                # Experience
                jd_min_exp = parse_required_experience(jd_text)
                if profile['experience_years'] >= jd_min_exp:
                    st.markdown(f"✅ **Experience:** Meets requirement ({profile['experience_years']} years vs. {jd_min_exp} required)")
                else:
                    st.markdown(f"⚠️ **Experience:** Below requirement ({profile['experience_years']} years vs. {jd_min_exp} required)")
                
                # Education
                jd_edu_level = parse_education_level(jd_text)
                if profile['education_level'] >= jd_edu_level:
                    st.markdown(f"✅ **Education:** Meets requirement ({profile['education_degree']})")
                else:
                    st.markdown(f"⚠️ **Education:** Below requirement ({profile['education_degree']})")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Areas for Improvement**")
                
                # Missing skills
                if selected_candidate['missing_skills']:
                    st.markdown("**Missing Skills:**")
                    for skill in selected_candidate['missing_skills']:
                        st.markdown(f"<span class='skill-tag missing-skill'>{skill}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("No missing skills identified.")
                
                # Score breakdown
                st.markdown("**Score Breakdown:**")
                st.markdown(f"- Overall Match: {score}%")
                st.markdown(f"- Similarity Score: {selected_candidate['similarity']:.2f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Attention heatmap (placeholder)
            st.markdown("#### Attention Heatmap")
            st.info("The attention heatmap would show which parts of the resume received more attention from the model. This requires a custom implementation of the attention mechanism.")
    else:
        st.info("ℹ️ Please process resumes first to view explainability.")

elif page == "Settings":
    st.markdown("### Application Settings")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Model Configuration</div>', unsafe_allow_html=True)
    
    st.markdown("Configure the matching algorithm and parameters used for candidate ranking.")
    
    st.markdown("**Matching Engine:**")
    st.markdown("""
    - **High-End Semantic Match (Recommended)**: Uses BERT embeddings for deep semantic understanding of job descriptions and resumes.
    - **TF-IDF Cosine Similarity**: Uses traditional TF-IDF vectorization for keyword-based matching.
    """)
    
    st.markdown("**Scoring Parameters:**")
    st.markdown("""
    - **Minimum Match Score**: Candidates below this score will be filtered out.
    - **Minimum Experience**: Candidates with less experience will be filtered out.
    - **Blind Screening**: Hides candidate identity to reduce bias in the screening process.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Advanced Options</div>', unsafe_allow_html=True)
    
    st.markdown("**Positional Encoding:**")
    st.markdown("Enable positional encoding to consider the order of information in resumes.")
    positional_encoding = st.checkbox("Enable Positional Encoding", value=False)
    
    st.markdown("**Self-Attention Model:**")
    st.markdown("Use a self-attention model for more sophisticated matching.")
    self_attention = st.checkbox("Enable Self-Attention Model", value=False)
    
    st.markdown('</div>', unsafe_allow_html=True)
