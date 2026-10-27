import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Spam Detector | Threat Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Premium Cybersecurity SaaS Custom Styling (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Theme & Deep Navy Cybersecurity Canvas */
    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.12) 0%, transparent 45%),
                    radial-gradient(circle at 90% 85%, rgba(6, 182, 212, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.05) 0%, transparent 60%),
                    #070b18 !important;
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }

    /* Headings font */
    h1, h2, h3, h4, .cyber-font {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.015em;
    }

    /* Top Executive Header */
    .cyber-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.7) 50%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 18px;
        padding: 1.75rem 2.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
    }

    .cyber-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #06b6d4, #6366f1, #a855f7, #ec4899);
    }

    .cyber-title {
        font-size: 2.35rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 0%, #38bdf8 55%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .cyber-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.35rem;
        margin-bottom: 0;
        font-weight: 400;
    }

    /* Pulsing Active Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.35rem 0.9rem;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        color: #34d399;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.8);
        animation: pulse-green 1.8s infinite;
    }

    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.8); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Executive Metrics Cards */
    .metric-card-cyber {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 14px;
        padding: 1.25rem 1.1rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        transition: transform 0.25s ease, border-color 0.25s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card-cyber:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
    }

    .metric-card-cyber::after {
        content: '';
        position: absolute;
        top: 0;
        left: 20%;
        right: 20%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #38bdf8, transparent);
    }

    .metric-number {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
    }

    .metric-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.06em;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.15);
        padding: 8px;
        border-radius: 14px;
        margin-bottom: 1.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        color: #94a3b8 !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 9px 18px !important;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0 !important;
        background: rgba(255, 255, 255, 0.04);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.28) 0%, rgba(6, 182, 212, 0.2) 100%) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.2);
    }

    /* Textarea Styling */
    .stTextArea textarea {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 14px !important;
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.98rem !important;
        padding: 1rem !important;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.5) !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25), inset 0 2px 8px rgba(0, 0, 0, 0.5) !important;
    }

    /* Gradient Primary Action Button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        border: none !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.35) !important;
        transition: all 0.3s ease !important;
    }

    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.55) !important;
        transform: translateY(-2px) !important;
    }

    /* Secondary Action Button */
    div.stButton > button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.6) !important;
        color: #94a3b8 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        color: #f8fafc !important;
        background: rgba(51, 65, 85, 0.8) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Prediction Result Cards */
    .threat-card-spam {
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.15) 0%, rgba(159, 18, 57, 0.28) 100%);
        border: 1px solid rgba(244, 63, 94, 0.5);
        border-radius: 16px;
        padding: 1.75rem;
        text-align: center;
        margin: 1.25rem 0;
        box-shadow: 0 10px 30px rgba(225, 29, 72, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(14px);
        animation: threat-pulse 2.5s infinite;
    }

    @keyframes threat-pulse {
        0% { border-color: rgba(244, 63, 94, 0.5); }
        50% { border-color: rgba(244, 63, 94, 0.85); box-shadow: 0 12px 35px rgba(225, 29, 72, 0.35); }
        100% { border-color: rgba(244, 63, 94, 0.5); }
    }

    .threat-card-ham {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(5, 150, 105, 0.24) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-radius: 16px;
        padding: 1.75rem;
        text-align: center;
        margin: 1.25rem 0;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(14px);
    }

    .threat-badge-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.015em;
        margin-bottom: 0.35rem;
    }

    .threat-title-spam {
        color: #fb7185;
        text-shadow: 0 0 25px rgba(251, 113, 133, 0.4);
    }

    .threat-title-ham {
        color: #34d399;
        text-shadow: 0 0 25px rgba(52, 211, 153, 0.4);
    }

    /* Probability Meter Box */
    .prob-meter-box {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem;
        margin-top: 1rem;
    }

    /* Token Chips */
    .token-chip-spam {
        display: inline-flex;
        align-items: center;
        padding: 0.28rem 0.65rem;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        background: rgba(244, 63, 94, 0.18);
        border: 1px solid rgba(244, 63, 94, 0.45);
        color: #fda4af;
        border-radius: 7px;
        margin: 0.25rem;
        box-shadow: 0 2px 6px rgba(244, 63, 94, 0.15);
    }

    .token-chip-ham {
        display: inline-flex;
        align-items: center;
        padding: 0.28rem 0.65rem;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #a7f3d0;
        border-radius: 7px;
        margin: 0.25rem;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.15);
    }

    /* Counter Badge */
    .counter-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.25rem 0.7rem;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        color: #94a3b8;
    }

    /* Cyber Callout & Viva Box */
    .cyber-info-box {
        background: rgba(15, 23, 42, 0.7);
        border-left: 4px solid #06b6d4;
        border-radius: 0 12px 12px 0;
        padding: 1.1rem 1.35rem;
        margin-bottom: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .cyber-warn-box {
        background: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        border-radius: 0 12px 12px 0;
        padding: 1.1rem 1.35rem;
        margin-bottom: 1rem;
        border-top: 1px solid rgba(245, 158, 11, 0.2);
        border-right: 1px solid rgba(245, 158, 11, 0.2);
        border-bottom: 1px solid rgba(245, 158, 11, 0.2);
    }

    /* Footer - Developer Credit Requirement */
    .app-footer {
        margin-top: 4rem;
        padding: 2rem 1rem 1.5rem 1rem;
        text-align: center;
        border-top: 1px solid rgba(56, 189, 248, 0.15);
        background: linear-gradient(180deg, transparent 0%, rgba(15, 23, 42, 0.5) 100%);
    }

    .app-footer-brand {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        color: #f1f5f9;
        margin-bottom: 0.3rem;
    }

    .app-footer-sub {
        font-size: 0.82rem;
        color: #64748b;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Dataset & Train Real ML Pipeline
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing ML Pipeline & Threat Corpus...")
def load_and_train_pipeline():
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    if not os.path.exists(dataset_path):
        dataset_path = "dataset.csv"

    df = pd.read_csv(dataset_path)

    # 80/20 train/test split with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'],
        df['label'],
        test_size=0.20,
        random_state=42,
        stratify=df['label']
    )

    # Real scikit-learn Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('nb', MultinomialNB(alpha=0.2))
    ])

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Calculate actual held-out metrics
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', pos_label='spam'
    )
    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])

    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'test_size': len(X_test),
        'train_size': len(X_train),
        'total_size': len(df),
        'spam_count': int((df['label'] == 'spam').sum()),
        'ham_count': int((df['label'] == 'ham').sum())
    }

    return pipeline, df, metrics, X_test, y_test

# Load Pipeline
try:
    pipeline, df_dataset, model_metrics, X_test_samples, y_test_samples = load_and_train_pipeline()
    ml_engine_ready = True
except Exception as e:
    st.error(f"Error loading threat corpus or training pipeline: {str(e)}")
    ml_engine_ready = False
    st.stop()

# ---------------------------------------------------------
# Helper: Feature Interpretability Breakdown
# ---------------------------------------------------------
def explain_text_features(text, pipeline):
    vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['nb']

    vec = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())

    nonzero_cols = vec.indices
    if len(nonzero_cols) == 0:
        return []

    ham_idx = list(classifier.classes_).index('ham')
    spam_idx = list(classifier.classes_).index('spam')

    token_info = []
    for idx in nonzero_cols:
        token = feature_names[idx]
        tfidf_val = vec[0, idx]
        spam_logprob = classifier.feature_log_prob_[spam_idx][idx]
        ham_logprob = classifier.feature_log_prob_[ham_idx][idx]
        diff = spam_logprob - ham_logprob  # Positive diff points to spam

        token_info.append({
            'token': token,
            'tfidf': round(float(tfidf_val), 3),
            'affinity': 'SPAM' if diff > 0 else 'NOT SPAM',
            'weight': round(float(diff), 3)
        })

    token_info.sort(key=lambda x: abs(x['weight']), reverse=True)
    return token_info

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
status_badge_html = """
<div class="status-badge">
    <span class="pulse-dot"></span> ML ENGINE ACTIVE
</div>
""" if ml_engine_ready else """
<div class="status-badge" style="background: rgba(239, 68, 68, 0.15); color: #f87171; border-color: rgba(239, 68, 68, 0.3);">
    ● ENGINE OFFLINE
</div>
"""

st.markdown(f"""
<div class="cyber-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div>
            <h1 class="cyber-title">🛡️ AI SPAM DETECTOR</h1>
            <p class="cyber-subtitle">Intelligent Email & SMS Threat Analysis • Natural Language Processing</p>
        </div>
        <div>
            {status_badge_html}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Executive Metrics Summary Dashboard (Real Dataset Stats)
# ---------------------------------------------------------
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(f"""
    <div class="metric-card-cyber">
        <div class="metric-number">{model_metrics['total_size']}</div>
        <div class="metric-title">📁 Total Corpus Messages</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card-cyber">
        <div class="metric-number" style="background: linear-gradient(135deg, #fb7185 0%, #f43f5e 100%); -webkit-background-clip: text;">{model_metrics['spam_count']}</div>
        <div class="metric-title">🚨 Spam Threat Samples</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card-cyber">
        <div class="metric-number" style="background: linear-gradient(135deg, #34d399 0%, #10b981 100%); -webkit-background-clip: text;">{model_metrics['ham_count']}</div>
        <div class="metric-title">✅ Legitimate Samples</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown("""
    <div class="metric-card-cyber">
        <div class="metric-number" style="font-size: 1.35rem; padding-top: 0.5rem; background: linear-gradient(135deg, #c084fc 0%, #818cf8 100%); -webkit-background-clip: text;">MultinomialNB</div>
        <div class="metric-title">⚙️ Core Model Algorithm</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Threat Feeds & Quick Samples
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Quick Attack & Normal Samples")
    st.caption("Click any preset payload to load into the analyzer:")

    quick_samples = [
        ("🚨 Bank Phishing Alert", "URGENT: Your Wells Fargo bank account has been temporarily locked due to suspicious activity. Verify your identity now at http://secure-wellsfargo-login.com"),
        ("🎁 Lottery Gift Card Scam", "CONGRATULATIONS! You have won a $1,000 Walmart Gift Card. Click here to claim your prize immediately: http://bit.ly/claim-walmart"),
        ("📦 DHL Pending Fee Fraud", "Your DHL parcel tracking #849104829 has delivery fees pending ($3.99). Pay now to prevent package return to sender: http://dhl-express-redelivery.org"),
        ("📚 College Assignment (Legit)", "Dear Students, the deadline for submitting the Machine Learning assignment has been extended to Friday 5 PM. Best regards, Prof. Sharma."),
        ("💻 Team Standup Notice (Legit)", "Good morning team, our daily standup is rescheduled to 11:30 AM today due to the client presentation. Please join via the Google Meet link."),
        ("✈️ Flight Boarding Alert (Legit)", "The flight AI-842 from Delhi to Mumbai has commenced boarding at Gate 14B. Passengers are requested to proceed with boarding passes.")
    ]

    for label, text in quick_samples:
        if st.button(label, use_container_width=True):
            st.session_state['input_message'] = text

    st.markdown("---")
    st.markdown("### ⚙️ Engine Specifications")
    st.markdown(f"""
    - **Classification Model**: `MultinomialNB`
    - **Laplace Smoothing (α)**: `0.2`
    - **Vectorization**: `TF-IDF (1, 2) n-grams`
    - **Feature Sublinear TF**: `Enabled`
    - **Held-Out Test Accuracy**: `{model_metrics['accuracy'] * 100:.1f}%`
    - **Held-Out Test Split**: `20% ({model_metrics['test_size']} samples)`
    """)

    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 0.85rem; font-size: 0.82rem; color: #94a3b8;">
        💡 <b>College Project Note:</b> Built for NLP demonstrations and college viva presentations. Real production systems incorporate SPF/DKIM headers, DNS blacklists, and neural language models.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Navigation Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🛡️ Live Threat Analysis",
    "📊 Model Performance & Metrics",
    "🗂️ Threat Intelligence Corpus",
    "🎓 College Viva & Defense Guide"
])

# ---------------------------------------------------------
# TAB 1: Live Threat Analysis
# ---------------------------------------------------------
with tab1:
    col_input, col_guide = st.columns([1.65, 1], gap="large")

    with col_input:
        st.markdown("### ✉️ Live Message Threat Analyzer")

        if 'input_message' not in st.session_state:
            st.session_state['input_message'] = ""

        # Current text & character count
        current_text = st.session_state.get('input_message', '')
        char_count = len(current_text)
        word_count = len(current_text.split()) if current_text.strip() else 0

        # Input header with character and word counters
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 500;">Enter Email Header, Body, or SMS Message:</span>
            <div class="counter-badge">
                <span>📝 {char_count} chars</span> • <span>🔤 {word_count} words</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        user_input = st.text_area(
            label="Message content to evaluate",
            value=st.session_state['input_message'],
            height=175,
            placeholder="Paste or type suspicious email text, promotional offers, or legitimate communications here...\n\nExample: 'URGENT: Click this link to confirm your account password or access will be revoked within 24 hours!'",
            label_visibility="collapsed"
        )

        col_action1, col_action2 = st.columns([1.3, 1])
        with col_action1:
            analyze_clicked = st.button("⚡ ANALYZE MESSAGE", type="primary", use_container_width=True)
        with col_action2:
            if st.button("🗑️ Clear Analyzer", use_container_width=True):
                st.session_state['input_message'] = ""
                st.rerun()

        # Prediction Processing
        if analyze_clicked:
            clean_text = user_input.strip()
            if not clean_text:
                st.warning("⚠️ Input buffer is empty. Please enter an email or SMS message to analyze.")
            else:
                pred_label = pipeline.predict([clean_text])[0]
                pred_proba = pipeline.predict_proba([clean_text])[0]
                classes = list(pipeline.classes_)
                spam_prob = pred_proba[classes.index('spam')] * 100
                ham_prob = pred_proba[classes.index('ham')] * 100

                # Display High-Impact Threat Banner
                if pred_label == 'spam':
                    st.markdown(f"""
                    <div class="threat-card-spam">
                        <div class="threat-badge-title threat-title-spam">🚨 SPAM DETECTED</div>
                        <p style="margin: 0; color: #fecdd3; font-size: 1.05rem;">
                            High risk: This message matches recognized patterns of spam, credential harvesting, or financial scams.
                        </p>
                        <div style="margin-top: 0.75rem; font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; color: #fb7185; font-weight: 700;">
                            Model Confidence Score: {spam_prob:.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="threat-card-ham">
                        <div class="threat-badge-title threat-title-ham">✅ NOT SPAM (LEGITIMATE)</div>
                        <p style="margin: 0; color: #a7f3d0; font-size: 1.05rem;">
                            Clean message: Linguistic structures align with standard legitimate, workplace, or institutional communication.
                        </p>
                        <div style="margin-top: 0.75rem; font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; color: #34d399; font-weight: 700;">
                            Model Confidence Score: {ham_prob:.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Professional Probability Meter
                st.markdown("""
                <div class="prob-meter-box">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                        <span style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8;">
                            📊 Posterior Probability Distribution
                        </span>
                        <span style="font-size: 0.75rem; color: #64748b;">
                            *Calculated via Multinomial Naive Bayes Bayes' Theorem
                        </span>
                    </div>
                """, unsafe_allow_html=True)

                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.25rem;">
                        <span style="color: #fb7185; font-weight: 600;">🚨 Spam Probability</span>
                        <span style="font-family: 'JetBrains Mono', monospace; color: #fb7185; font-weight: 700;">{spam_prob:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(spam_prob / 100.0))

                with p_col2:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.25rem;">
                        <span style="color: #34d399; font-weight: 600;">✅ Legitimate Probability</span>
                        <span style="font-family: 'JetBrains Mono', monospace; color: #34d399; font-weight: 700;">{ham_prob:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(ham_prob / 100.0))

                st.markdown("</div>", unsafe_allow_html=True)

                # Feature Interpretability & Model Token Attribution
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                st.markdown("#### 🔬 Model Interpretability: Key Token Attribution")
                tokens_info = explain_text_features(clean_text, pipeline)

                if tokens_info:
                    st.caption("Vocabulary terms present in your input with their TF-IDF weight and mathematical class affinity (log-likelihood difference):")
                    
                    spam_tokens = [t for t in tokens_info if t['affinity'] == 'SPAM']
                    ham_tokens = [t for t in tokens_info if t['affinity'] == 'NOT SPAM']

                    t_col1, t_col2 = st.columns(2)
                    with t_col1:
                        st.markdown("**Tokens Leaning Toward SPAM:**")
                        if spam_tokens:
                            chips_html = "".join([
                                f'<span class="token-chip-spam">{t["token"]} <span style="opacity:0.75; font-size:0.72rem; margin-left:4px;">(TF-IDF: {t["tfidf"]})</span></span>' 
                                for t in spam_tokens
                            ])
                            st.markdown(chips_html, unsafe_allow_html=True)
                        else:
                            st.write("_No spam-affiliated vocabulary tokens identified._")

                    with t_col2:
                        st.markdown("**Tokens Leaning Toward LEGITIMATE:**")
                        if ham_tokens:
                            chips_html = "".join([
                                f'<span class="token-chip-ham">{t["token"]} <span style="opacity:0.75; font-size:0.72rem; margin-left:4px;">(TF-IDF: {t["tfidf"]})</span></span>' 
                                for t in ham_tokens
                            ])
                            st.markdown(chips_html, unsafe_allow_html=True)
                        else:
                            st.write("_No legitimate-affiliated vocabulary tokens identified._")
                else:
                    st.info("ℹ️ Out-of-Vocabulary Notice: None of the terms in this message exist in the training vocabulary. Multinomial Naive Bayes falls back to the prior probability distribution.")

    with col_guide:
        st.markdown("### 💡 Analysis Architecture")
        st.markdown("""
        <div class="cyber-info-box">
            <b>1. Feature Vectorization (TF-IDF)</b><br>
            <span style="font-size: 0.88rem; color: #94a3b8;">
                Converts raw unstructured text into high-dimensional numerical vectors, penalizing ubiquitous generic terms while rewarding rare, high-signal tokens.
            </span>
        </div>
        <div class="cyber-info-box">
            <b>2. Multinomial Naive Bayes</b><br>
            <span style="font-size: 0.88rem; color: #94a3b8;">
                Applies Bayes' theorem with conditional feature independence to compute posterior likelihoods across classes:
                <br><code>P(C | Text) ∝ P(C) ∏ P(w_i | C)</code>
            </span>
        </div>
        <div class="cyber-info-box">
            <b>3. Log-Likelihood Decision</b><br>
            <span style="font-size: 0.88rem; color: #94a3b8;">
                Sums log probabilities to avoid floating point underflow and selects the class corresponding to the highest posterior probability.
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="cyber-warn-box">
            <b style="color: #fbbf24;">⚠️ Academic Demonstration Limitations</b><br>
            <span style="font-size: 0.84rem; color: #cbd5e1;">
                This system operates on a curated demonstration corpus of 100 samples. Production spam filters (e.g., Google Workspace, Microsoft 365) employ billions of data points, SPF/DKIM verification, and transformer neural networks.
            </span>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: Model Performance & Metrics
# ---------------------------------------------------------
with tab2:
    st.markdown("### 📊 Held-Out Test Evaluation")
    st.caption("All evaluation metrics below were calculated from a strictly separated 20% test partition (20 samples) completely withheld from training.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card-cyber">
            <div class="metric-number">{model_metrics['accuracy'] * 100:.1f}%</div>
            <div class="metric-title">🎯 Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card-cyber">
            <div class="metric-number">{model_metrics['precision'] * 100:.1f}%</div>
            <div class="metric-title">🔍 Precision (Spam)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card-cyber">
            <div class="metric-number">{model_metrics['recall'] * 100:.1f}%</div>
            <div class="metric-title">📡 Recall (Spam)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card-cyber">
            <div class="metric-number">{model_metrics['f1'] * 100:.1f}%</div>
            <div class="metric-title">⚖️ F1-Score (Spam)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    col_cm_left, col_cm_right = st.columns([1.1, 1], gap="large")

    with col_cm_left:
        st.markdown("#### 🎯 Confusion Matrix")
        cm = model_metrics['confusion_matrix']
        cm_df = pd.DataFrame(
            cm,
            index=['Actual: NOT SPAM', 'Actual: SPAM'],
            columns=['Predicted: NOT SPAM', 'Predicted: SPAM']
        )
        st.table(cm_df)

        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1rem; font-size: 0.9rem;">
            <b>Matrix Breakdown:</b>
            <ul style="margin: 0.5rem 0 0 0; padding-left: 1.25rem; color: #94a3b8;">
                <li><b>True Negatives (TN)</b>: <code>{cm[0][0]}</code> — Legitimate messages accurately allowed.</li>
                <li><b>False Positives (FP)</b>: <code>{cm[0][1]}</code> — Legitimate messages wrongly flagged as spam.</li>
                <li><b>False Negatives (FN)</b>: <code>{cm[1][0]}</code> — Spam messages that evaded detection.</li>
                <li><b>True Positives (TP)</b>: <code>{cm[1][1]}</code> — Spam threats accurately intercepted.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_cm_right:
        st.markdown("#### ⚖️ Classification Trade-off Analysis")
        st.markdown("""
        In cybersecurity and email protection, **False Positives (FP)** are typically more catastrophic than **False Negatives (FN)**:
        - If a spam email bypasses the filter (**False Negative**), the user simply deletes it.
        - If an urgent university admission letter or salary notification is falsely trapped in spam (**False Positive**), critical communication is lost.
        
        Therefore, real-world spam classifiers adjust the decision threshold upwards (e.g. $P(\\text{Spam}) > 0.75$) to drastically minimize False Positives.
        """)

# ---------------------------------------------------------
# TAB 3: Threat Intelligence Corpus (Dataset Explorer)
# ---------------------------------------------------------
with tab3:
    st.markdown("### 🗂️ Threat Intelligence Corpus Explorer")
    st.caption("Inspect and filter the curated training messages used to build the NLP model.")

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        search_term = st.text_input("🔍 Search message text by keyword", placeholder="e.g. bank, lottery, assignment, package...", label_visibility="collapsed")
    with col_s2:
        class_filter = st.selectbox("Filter by Category", ["All Categories", "Spam Threats Only", "Legitimate Messages Only"])

    # Filter logic
    display_df = df_dataset.copy()
    if class_filter == "Spam Threats Only":
        display_df = display_df[display_df['label'] == 'spam']
    elif class_filter == "Legitimate Messages Only":
        display_df = display_df[display_df['label'] == 'ham']

    if search_term:
        display_df = display_df[display_df['message'].str.contains(search_term, case=False, na=False)]

    st.markdown(f"**Showing {len(display_df)} matching records** out of {len(df_dataset)} total messages:")

    # Add visual badge column
    styled_df = display_df.copy()
    styled_df['Status'] = styled_df['label'].apply(lambda x: "🚨 SPAM" if x == 'spam' else "✅ HAM")
    styled_df = styled_df[['Status', 'message', 'label']]

    st.dataframe(
        styled_df[['Status', 'message']],
        use_container_width=True,
        height=320,
        column_config={
            "Status": st.column_config.TextColumn("Classification", width="small"),
            "message": st.column_config.TextColumn("Message Content", width="large")
        }
    )

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    c_graph1, c_graph2 = st.columns([1, 1])
    with c_graph1:
        st.markdown("#### 📈 Class Balance")
        st.bar_chart(df_dataset['label'].value_counts())
    with c_graph2:
        st.markdown("#### 📝 Message Length Profile")
        df_dataset['length'] = df_dataset['message'].apply(len)
        st.line_chart(df_dataset.groupby('label')['length'].mean())

# ---------------------------------------------------------
# TAB 4: College Viva & Defense Guide
# ---------------------------------------------------------
with tab4:
    st.markdown("### 🎓 College Viva & Technical Defense Guide")
    st.caption("Conceptual explanations, mathematical foundations, and answers for academic evaluations.")

    with st.expander("1. What is TF-IDF and how does it improve upon Bag-of-Words (BoW)?", expanded=True):
        st.markdown("""
        **Bag of Words (BoW)** merely counts word frequencies across a document. However, high-frequency grammatical glue words (*e.g., 'is', 'the', 'at', 'with'*) dominate without providing discriminative value between spam and legitimate communications.

        **TF-IDF (Term Frequency - Inverse Document Frequency)** resolves this by weighting words according to their corpus-wide specificity:

        - **Term Frequency (TF)**:
          $$\\text{TF}(t, d) = \\frac{\\text{Frequency of term } t \\text{ in document } d}{\\text{Total words in document } d}$$

        - **Inverse Document Frequency (IDF)**:
          $$\\text{IDF}(t) = \\log\\left(\\frac{1 + N}{1 + \\text{DF}(t)}\\right) + 1$$
          *(Where $N$ is the total document count and $\\text{DF}(t)$ is the count of documents containing term $t$.)*

        - **Composite TF-IDF Score**:
          $$\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\text{IDF}(t)$$

        This mathematical formulation rewards words that occur frequently in a specific message but rarely in the corpus at large (*e.g., 'wire transfer', 'crypto', 'giftcard'*).
        """)

    with st.expander("2. What is Multinomial Naive Bayes and why is it termed 'Naive'?", expanded=False):
        st.markdown("""
        **Multinomial Naive Bayes** is a probabilistic classifier based on **Bayes' Theorem**:

        $$P(C \\mid X) = \\frac{P(X \\mid C) \\cdot P(C)}{P(X)}$$

        Where:
        - $P(C \\mid X)$ is the **Posterior probability** of class $C$ (Spam or Ham) given the observed text $X$.
        - $P(X \\mid C)$ is the **Likelihood** of features $X$ given class $C$.
        - $P(C)$ is the **Prior probability** of class $C$ (computed from class distribution).
        - $P(X)$ is the **Evidence** normalization constant.

        ### Why is it termed 'Naive'?
        It makes the strong (naive) assumption of **conditional independence**: it presumes that all words in an email appear completely independently of one another given the category:
        $$P(x_1, x_2, \\dots, x_n \\mid C) = \\prod_{i=1}^n P(x_i \\mid C)$$

        Even though natural language violates this assumption (*e.g., 'gift' and 'card' frequently co-occur*), in practice Naive Bayes performs remarkably well for text categorization because relative ranking is preserved.
        """)

    with st.expander("3. What is Laplace Smoothing and why is alpha (α) indispensable?", expanded=False):
        st.markdown("""
        If a new, unseen email contains a term that was never present in the spam training subset, its estimated probability would be zero:
        $$P(w_{\\text{new}} \\mid \\text{Spam}) = 0$$

        Because Naive Bayes multiplies all individual feature probabilities:
        $$P(X \\mid \\text{Spam}) = P(w_1 \\mid \\text{Spam}) \\times P(w_2 \\mid \\text{Spam}) \\times \\dots \\times 0 = 0$$

        A single unobserved word would reduce the entire posterior probability to zero, blinding the model. This is called the **Zero-Frequency Problem**.

        **Laplace Smoothing** solves this by adding a small positive pseudo-count $\\alpha$ (we use $\\alpha = 0.2$):
        $$P(w_i \\mid C) = \\frac{N_{ci} + \\alpha}{N_c + \\alpha \\cdot |V|}$$
        Where $|V|$ represents the total vocabulary size. This guarantees that all terms have a non-zero probability.
        """)

    with st.expander("4. Step-by-Step Training and Inference Workflow", expanded=False):
        st.markdown("""
        ```
        Raw Email / SMS Message Corpus
                    ↓
        [ Text Preprocessing ]      --> Lowercasing, stop-word elimination, n-gram extraction (1-2)
                    ↓
        [ TF-IDF Vectorization ]    --> Converts text tokens into sparse TF-IDF feature matrices
                    ↓
        [ Train/Test Split ]        --> 80% Training partition, 20% Held-out evaluation partition
                    ↓
        [ Model Training ]          --> MultinomialNB fits class priors & feature log probabilities
                    ↓
        [ Inference & Decision ]    --> New Text → TF-IDF Transform → Compute Log-likelihoods → Output Prediction & Probability
        ```
        """)

    with st.expander("5. Viva Examination Q&A Cheat Sheet", expanded=False):
        st.markdown("""
        - **Q: Why use Multinomial Naive Bayes instead of Gaussian Naive Bayes for text?**
          *A:* Gaussian Naive Bayes assumes features follow a continuous Gaussian bell curve. Multinomial Naive Bayes models discrete term counts / frequency distributions, which matches text frequency data.
        
        - **Q: How can spammers attempt to bypass this model?**
          *A:* Through adversarial techniques such as character obfuscation (*e.g., 'V1AGRA' instead of 'Viagra'*), adding zero-width spaces, embedding text inside images, or appending legitimate text ("good word attacks") to dilute spam term frequencies.

        - **Q: What additions are required for a real production mail filter?**
          *A:* Domain email validation (SPF, DKIM, DMARC), real-time DNS blocklists (RBLs), sender IP reputation scores, attachment malware scanning, and fine-tuned Transformer models (BERT/RoBERTa).
        """)

# ---------------------------------------------------------
# Footer - Developer Credit Requirement
# ---------------------------------------------------------
st.markdown("""
<div class="app-footer">
    <div class="app-footer-brand">Developed by Prathmesh Khedkar</div>
    <div class="app-footer-sub">AI Spam Detector • Cybersecurity & Machine Learning Demonstration Dashboard</div>
</div>
""", unsafe_allow_html=True)
