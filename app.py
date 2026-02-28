import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, random, json, difflib, joblib, time
import plotly.graph_objects as go
from fpdf import FPDF
import numpy as np
import cv2
from yoga_utils import YogaCoach
import streamlit as st
import os
import requests

# --- AUTO-DOWNLOAD LARGE MODEL FROM DRIVE ---
MODEL_PATH = "disease_model.pkl"
# Your specific File ID from the link you provided
GOOGLE_DRIVE_ID = "16KrLnRvQfezXUk0zPi1wxage8F1erSk4"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_ID}"
# --- APP CONFIG & INITIALIZATION ---



@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI Model from Cloud... Please wait."):
            response = requests.get(DOWNLOAD_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model Downloaded Successfully!")
            else:
                st.error("Failed to download model from Google Drive. Check link permissions!")

download_model() # Run the downloader
st.set_page_config(page_title="AI Yoga Guru", layout="wide", page_icon="🧘")

# Initialize the AI Coach from our utils file
@st.cache_resource
def init_coach():
    return YogaCoach()

coach = init_coach()

# Session State to prevent app refresh during camera use
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'top_recs' not in st.session_state:
    st.session_state.top_recs = None

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
    except:
        metadata = {"pose_image_mapping": {}, "target_area_mapping": {}}
    
    # Load the model trained in Google Colab
    model_path = 'disease_model.pkl'
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    return metadata, model

metadata, disease_model = load_assets()
pose_map = metadata.get('pose_image_mapping', {})
target_map = metadata.get('target_area_mapping', {})

# --- DATA LOADING ---
@st.cache_data
def load_data():
    u_path = r"C:\Users\ragha\OneDrive\Desktop\yoga recommendation\Kaggle Health & Lifestyle Dataset\health_lifestyle_dataset.csv"
    y_path = r"C:\Users\ragha\OneDrive\Desktop\yoga recommendation\Kaggle Yoga Poses Recommendation\Yoga Data.xlsx"
    
    df_u = pd.read_csv(u_path)
    df_y = pd.read_excel(y_path)
    
    # Clean and combine tags for NLP recommendation
    text_cols = ['Benefits', 'Targeted Mental Problems', 'Targeted Physical Problems']
    for col in text_cols:
        df_y[col] = df_y[col].astype(str).replace('nan', '')
    df_y['tags'] = df_y[text_cols].agg(' '.join, axis=1)
    
    return df_u, df_y

df_user, df_yoga = load_data()

# --- NLP ENGINE ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_yoga['tags'])

# --- HELPER FUNCTIONS ---
def get_image(asana_name):
    base_path = r"C:\Users\ragha\OneDrive\Desktop\yoga recommendation\Kaggle Yoga Pose Classification"
    search_term = pose_map.get(asana_name, asana_name)
    try:
        all_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        matches = difflib.get_close_matches(search_term, all_folders, n=1, cutoff=0.3)
        if matches:
            folder = matches[0]
            internal = folder.split('-202')[0] 
            img_dir = os.path.join(base_path, folder, internal)
            imgs = [i for i in os.listdir(img_dir) if i.lower().endswith(('.jpg', '.png', '.jpeg'))]
            return os.path.join(img_dir, random.choice(imgs)) if imgs else None
    except:
        return None

def generate_pdf(user, recs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "AI Yoga Guru: Your Routine", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"BMI: {user['bmi']} | Risk: {user['disease_risk']}", ln=True)
    pdf.ln(10)
    for _, row in recs.iterrows():
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, f"Pose: {row['AName']}", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 10, f"Benefits: {row['Benefits']}")
        pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')

# --- SIDEBAR ---
st.sidebar.title("🧘 AI Yoga Guru")
mode = st.sidebar.radio("Profile Type", ["Existing User", "New User Profile"])
lvl = st.sidebar.selectbox("Difficulty Level", ["All Levels", 1, 2, 3])

if mode == "Existing User":
    u_id = st.sidebar.number_input("Enter User ID", min_value=1, value=101)
    if st.sidebar.button("Fetch Analysis"):
        st.session_state.current_user = df_user[df_user['id'] == u_id].iloc[0].to_dict()
        st.session_state.top_recs = None
else:
    u_bmi = st.sidebar.slider("BMI", 10.0, 50.0, 24.0)
    u_sys = st.sidebar.slider("Systolic BP", 90, 200, 120)
    if st.sidebar.button("Generate My Routine"):
        st.session_state.current_user = {
            'bmi': u_bmi, 
            'systolic_bp': u_sys, 
            'diastolic_bp': 80, 
            'sleep_hours': 7, 
            'disease_risk': "Analyzing..."
        }
        st.session_state.top_recs = None

# --- MAIN DASHBOARD ---
if st.session_state.current_user:
    u = st.session_state.current_user
    
    # AI Disease Prediction
    if disease_model and u.get('disease_risk') == "Analyzing...":
        u['disease_risk'] = disease_model.predict([[u['bmi'], u['systolic_bp'], u['diastolic_bp'], u['sleep_hours']]])[0]

    st.header("Health Dashboard")
    
    # Visual BMI Gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = u['bmi'], 
        title = {'text': "BMI Category"},
        gauge = {'axis': {'range': [10, 50]}, 'steps' : [
            {'range': [10, 18.5], 'color': "lightblue"},
            {'range': [18.5, 25], 'color': "green"},
            {'range': [25, 30], 'color': "yellow"},
            {'range': [30, 50], 'color': "red"}]}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Recommendation Engine
    if st.session_state.top_recs is None:
        q = f"bmi {u['bmi']} blood pressure {u['systolic_bp']}"
        vec = tfidf.transform([q])
        df_yoga['score'] = cosine_similarity(vec, tfidf_matrix).flatten()
        res = df_yoga.copy()
        if lvl != "All Levels":
            res = res[res['Level'] == lvl]
        st.session_state.top_recs = res.sort_values('score', ascending=False).head(5)

    st.subheader("🧘 Personalized Recommendations")
    for _, row in st.session_state.top_recs.iterrows():
        with st.expander(f"{row['AName']} (Level {row['Level']})"):
            c1, c2 = st.columns([1, 2])
            img_p = get_image(row['AName'])
            if img_p: c1.image(img_p, use_container_width=True)
            c2.write(f"**Target:** {target_map.get(str(row['Target Areas']), 'Full Body')}")
            c2.write(f"**Benefits:** {row['Benefits']}")
            if str(row['Contraindications']).lower() not in ["none", "nan"]:
                c2.error(f"**Safety:** {row['Contraindications']}")

    # PDF Download
    st.divider()
    pdf_data = generate_pdf(u, st.session_state.top_recs)
    st.download_button("📥 Download Plan as PDF", data=pdf_data, file_name="yoga_plan.pdf")

    # Live AI Coach
    st.divider()
    st.subheader("📸 AI Live Pose Correction")
    pose_name = st.selectbox("Select Pose to Verify", st.session_state.top_recs['AName'].tolist())
    
    timer_placeholder = st.empty()
    if st.button("🚀 Start 5s Timer & Capture"):
        for i in range(5, 0, -1):
            timer_placeholder.metric("Get Ready!", f"{i}s")
            time.sleep(1)
        timer_placeholder.empty()
        
        photo = st.camera_input("Hold your pose!")
        if photo:
            bytes_data = photo.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            
            message, annotated_img = coach.analyze_frame(rgb_img, pose_name)
            st.image(annotated_img, caption="AI Joint Tracking")
            st.info(message)
else:
    st.info("👈 Please enter a User ID or create a profile in the sidebar.")

