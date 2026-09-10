"""
app.py - MBG Sentiment Analyzer
Streamlit application for sentiment analysis of public comments
about Indonesia's Makan Bergizi Gratis (MBG) program.

Model: IndoBERT-base-p1 (BertForSequenceClassification, 3 classes)
Pipeline: Stream B preprocessing (slang normalization + BERT tokenization)
Label mapping: {0: Positive, 1: Neutral, 2: Negative}
"""

import logging
import time
from pathlib import Path
from typing import Dict

import streamlit as st
import streamlit.components.v1 as components
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SLANG_PATH = MODEL_DIR / "slang_dict_mbg.json"
MAX_INPUT_CHARS = 1000

SENTIMENT_CONFIG = {
    "Positive": {
        "color": "#22c55e",
        "bg_color": "rgba(34, 197, 94, 0.08)",
        "border_color": "rgba(34, 197, 94, 0.3)",
        "badge_bg": "rgba(34, 197, 94, 0.18)",
        "label_id": "SENTIMEN POSITIF",
        "description": "Komentar ini diprediksi mengandung sentimen <strong>positif</strong> &mdash; menunjukkan dukungan, apresiasi, atau pandangan optimis terhadap program MBG.",
    },
    "Neutral": {
        "color": "#38bdf8",
        "bg_color": "rgba(56, 189, 248, 0.08)",
        "border_color": "rgba(56, 189, 248, 0.3)",
        "badge_bg": "rgba(56, 189, 248, 0.18)",
        "label_id": "SENTIMEN NETRAL",
        "description": "Komentar ini diprediksi bersifat <strong>netral</strong> &mdash; menyampaikan informasi atau pandangan seimbang tanpa kecenderungan positif maupun negatif yang dominan.",
    },
    "Negative": {
        "color": "#f87171",
        "bg_color": "rgba(248, 113, 113, 0.08)",
        "border_color": "rgba(248, 113, 113, 0.3)",
        "badge_bg": "rgba(248, 113, 113, 0.18)",
        "label_id": "SENTIMEN NEGATIF",
        "description": "Komentar ini diprediksi mengandung sentimen <strong>negatif</strong> &mdash; menunjukkan kritik, kekecewaan, atau catatan terhadap pelaksanaan program MBG.",
    },
}

SAMPLE_COMMENTS = [
    {
        "category": "Positif",
        "text": "Alhamdulillah program MBG sangat bermanfaat dan membantu gizi anak sekolah di pelosok. Menunya bergizi, sehat, dan anak-anak senang sekali. Terima kasih pemerintah.",
    },
    {
        "category": "Netral",
        "text": "Pemerintah menjadwalkan uji coba distribusi program makan bergizi gratis di 50 sekolah wilayah Jawa Barat mulai pekan depan.",
    },
    {
        "category": "Negatif",
        "text": "Program MBG ini parah banget pelaksanaannya, makanannya basi dan anggarannya malah dikorupsi oknum! Sangat mengecewakan dan merugikan rakyat kecil.",
    },
    {
        "category": "Ambigu / Sarkasme",
        "text": "Katanya program MBG anggarannya triliunan buat rakyat kecil, tapi kok lauknya cuma tempe seiprit doang? Mantap banget deh solusinya.",
    },
]

st.set_page_config(
    page_title="MBG Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "MBG Sentiment Analyzer — Analisis sentimen komentar masyarakat terhadap program Makan Bergizi Gratis."},
)


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
    }
    
    .stApp { 
        background-color: #0b0f19 !important; 
        color: #f1f5f9 !important;
    }

    #MainMenu, footer { visibility: hidden; }

    /* Header & Sidebar Re-open Control */
    header[data-testid="stHeader"] { 
        background: transparent !important;
        height: 3.5rem !important;
        z-index: 99 !important;
    }
    
    [data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        color: #cbd5e1 !important;
        background: #131b2e !important;
        border: 1px solid #24324d !important;
        border-radius: 8px !important;
        margin-left: 0.75rem !important;
        margin-top: 0.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover {
        color: #ffffff !important;
        border-color: #3b82f6 !important;
        background: #1e3a5f !important;
    }
    [data-testid="stSidebarCollapsedControl"] button {
        color: inherit !important;
    }

    .block-container { 
        padding-top: 1.5rem !important; 
        padding-bottom: 3rem !important; 
        max-width: 920px !important; 
    }

    /* Top Navigasi Capsule (like reference design) */
    .nav-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: inline-flex !important;
        flex-wrap: wrap !important;
        background: #131b2e !important;
        border: 1px solid #24324d !important;
        border-radius: 100px !important;
        padding: 5px 8px !important;
        gap: 6px !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: transparent !important;
        padding: 6px 16px !important;
        border-radius: 100px !important;
        margin: 0 !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        display: inline-flex !important;
        align-items: center !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background: rgba(255, 255, 255, 0.06) !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background: #1e3a5f !important;
        border: 1px solid #3b82f6 !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] p {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Container Card Wrapper */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #131b2e !important;
        border: 1px solid #1e293b !important;
        border-radius: 14px !important;
        padding: 1.25rem 1.5rem !important;
        margin-bottom: 1.25rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f2c59 100%);
        border: 1px solid #1e3a8a;
        border-radius: 18px; 
        padding: 2.5rem 2.75rem; 
        margin-bottom: 1.5rem;
        position: relative; 
        overflow: hidden;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4);
    }
    .hero-container::before {
        content: ''; 
        position: absolute; 
        top: -50%; 
        right: -15%; 
        width: 60%; 
        height: 200%;
        background: radial-gradient(circle, rgba(59,130,246,0.18) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex; 
        align-items: center; 
        gap: 6px;
        background: rgba(59,130,246,0.15); 
        border: 1px solid rgba(59,130,246,0.35);
        color: #93c5fd; 
        font-size: 0.72rem; 
        font-weight: 700;
        letter-spacing: 0.1em; 
        padding: 4px 12px; 
        border-radius: 100px;
        margin-bottom: 0.9rem; 
        text-transform: uppercase;
    }
    .hero-badge::before {
        content: '';
        width: 6px;
        height: 6px;
        background: #38bdf8;
        border-radius: 50%;
        box-shadow: 0 0 8px #38bdf8;
    }
    .hero-title { 
        font-size: 2.2rem; 
        font-weight: 800; 
        color: #ffffff; 
        margin: 0 0 0.4rem 0; 
        line-height: 1.2; 
        letter-spacing: -0.02em; 
    }
    .hero-title span { 
        background: linear-gradient(90deg, #60a5fa, #a78bfa); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        background-clip: text; 
    }
    .hero-subtitle { 
        font-size: 0.95rem; 
        color: #94a3b8; 
        margin: 0 0 1.6rem 0; 
        line-height: 1.6; 
        max-width: 580px; 
    }
    .hero-stats { 
        display: flex; 
        gap: 2.2rem; 
        flex-wrap: wrap; 
    }
    .hero-stat { 
        display: flex; 
        flex-direction: column; 
        gap: 2px; 
    }
    .hero-stat-value { 
        font-size: 1.35rem; 
        font-weight: 700; 
        color: #f8fafc; 
    }
    .hero-stat-label { 
        font-size: 0.72rem; 
        color: #64748b; 
        text-transform: uppercase; 
        letter-spacing: 0.06em; 
        font-weight: 600;
    }

    .section-title { 
        font-size: 0.8rem; 
        font-weight: 700; 
        color: #94a3b8; 
        text-transform: uppercase; 
        letter-spacing: 0.08em; 
        margin-bottom: 0.9rem; 
    }

    /* Text Area */
    .stTextArea textarea { 
        font-family: 'Inter', sans-serif !important; 
        font-size: 0.95rem !important; 
        color: #f8fafc !important; 
        border-radius: 10px !important; 
        border: 1px solid #24324d !important; 
        background: #0f172a !important; 
        padding: 14px !important; 
        line-height: 1.65 !important; 
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important; 
    }
    .stTextArea textarea:focus { 
        border-color: #3b82f6 !important; 
        box-shadow: 0 0 0 3px rgba(59,130,246,0.25) !important; 
        outline: none !important; 
        background: #111b33 !important; 
    }
    .stTextArea textarea::placeholder {
        color: #475569 !important;
    }

    /* All Buttons */
    .stButton > button { 
        font-family: 'Inter', sans-serif !important; 
        font-weight: 600 !important; 
        font-size: 0.88rem !important; 
        border-radius: 9px !important; 
        padding: 0.55rem 1.25rem !important; 
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important; 
    }
    .stButton > button:hover { 
        background-color: #2e3d55 !important; 
        color: #ffffff !important; 
        border-color: #64748b !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    .stButton > button:active, .stButton > button:focus {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }

    /* Primary Button (Analisis Sentimen) */
    .stButton > button[kind="primary"], 
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover, 
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        color: #ffffff !important;
        border-color: #60a5fa !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.5) !important;
    }

    /* Secondary Button (Bersihkan) */
    .stButton > button[kind="secondary"], 
    .stButton > button[data-testid="stBaseButton-secondary"] {
        background-color: #1e293b !important;
        color: #cbd5e1 !important;
        border: 1px solid #334155 !important;
    }
    .stButton > button[kind="secondary"]:hover, 
    .stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #334155 !important;
        color: #ffffff !important;
        border-color: #64748b !important;
    }

    /* Tooltip styling */
    [data-baseweb="tooltip"], [data-baseweb="popover"], [data-testid="stTooltipContent"] {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
    }

    /* Results */
    .result-container { 
        border-radius: 14px; 
        padding: 1.75rem; 
        border: 1.5px solid; 
        margin-bottom: 1.25rem;
    }
    .result-header { 
        display: flex; 
        align-items: center; 
        justify-content: space-between; 
        margin-bottom: 1.25rem; 
    }
    .result-badge { 
        display: inline-flex; 
        align-items: center; 
        font-size: 0.72rem; 
        font-weight: 700; 
        letter-spacing: 0.1em; 
        padding: 4px 12px; 
        border-radius: 100px; 
        text-transform: uppercase; 
    }
    .result-label { 
        font-size: 1.8rem; 
        font-weight: 800; 
        letter-spacing: -0.01em; 
        margin: 6px 0 0 0; 
    }
    .confidence-block { 
        background: rgba(15, 23, 42, 0.6); 
        border-radius: 10px; 
        padding: 0.9rem 1.25rem; 
        margin-bottom: 1rem; 
        border: 1px solid rgba(255,255,255,0.06); 
    }
    .confidence-label { 
        font-size: 0.72rem; 
        font-weight: 600; 
        text-transform: uppercase; 
        letter-spacing: 0.08em; 
        color: #94a3b8; 
        margin-bottom: 4px; 
    }
    .confidence-value { 
        font-size: 2rem; 
        font-weight: 800; 
        line-height: 1.1; 
        letter-spacing: -0.02em; 
    }
    .result-description { 
        font-size: 0.88rem; 
        line-height: 1.65; 
        color: #cbd5e1; 
        margin-top: 0.75rem; 
    }

    /* Probability Bars */
    .prob-section { 
        margin-top: 1.25rem; 
        background: #0f172a; 
        border: 1px solid #1e293b; 
        border-radius: 10px; 
        padding: 1.25rem 1.5rem;
    }
    .prob-row { 
        display: flex; 
        align-items: center; 
        gap: 12px; 
        margin-bottom: 12px; 
    }
    .prob-row:last-child {
        margin-bottom: 0;
    }
    .prob-label { 
        font-size: 0.82rem; 
        font-weight: 600; 
        color: #cbd5e1; 
        min-width: 85px; 
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .prob-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .prob-bar-wrap { 
        flex: 1; 
        height: 8px; 
        background: #1e293b; 
        border-radius: 100px; 
        overflow: hidden; 
    }
    .prob-bar-fill { 
        height: 100%; 
        border-radius: 100px; 
        transition: width 0.5s ease; 
    }
    .prob-value { 
        font-size: 0.82rem; 
        font-weight: 700; 
        color: #f1f5f9; 
        min-width: 48px; 
        text-align: right; 
        font-variant-numeric: tabular-nums;
    }

    /* History */
    .history-item { 
        display: flex; 
        align-items: center; 
        justify-content: space-between; 
        padding: 9px 12px; 
        border-radius: 8px; 
        background: #1e293b; 
        border: 1px solid #334155; 
        margin-bottom: 8px; 
    }
    .history-text { 
        font-size: 0.8rem; 
        color: #cbd5e1; 
        white-space: nowrap; 
        overflow: hidden; 
        text-overflow: ellipsis; 
        max-width: 170px; 
    }
    .history-badge { 
        font-size: 0.7rem; 
        font-weight: 700; 
        padding: 3px 8px; 
        border-radius: 100px; 
        white-space: nowrap; 
    }

    /* Sidebar */
    [data-testid="stSidebar"] { 
        background: #090d16 !important; 
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] .stMarkdown { 
        color: #cbd5e1; 
    }
    .sidebar-logo { 
        display: flex; 
        flex-direction: column;
        gap: 3px;
        padding: 0.5rem 0 1.25rem 0; 
        border-bottom: 1px solid #1e293b; 
        margin-bottom: 1.25rem; 
    }
    .sidebar-logo-text { 
        font-size: 1.15rem; 
        font-weight: 800; 
        color: #f8fafc; 
        letter-spacing: -0.01em;
    }
    .sidebar-logo-sub { 
        font-size: 0.72rem; 
        color: #64748b; 
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .sidebar-section-title { 
        font-size: 0.68rem; 
        font-weight: 700; 
        color: #64748b; 
        text-transform: uppercase; 
        letter-spacing: 0.1em; 
        margin: 1.3rem 0 0.65rem 0; 
    }
    .sidebar-model-chip { 
        background: #131b2e; 
        border: 1px solid #1e293b; 
        border-radius: 8px; 
        padding: 8px 12px; 
        margin-bottom: 7px; 
    }
    .chip-label { 
        font-size: 0.68rem; 
        color: #64748b; 
        text-transform: uppercase; 
        letter-spacing: 0.06em; 
        margin-bottom: 2px; 
        font-weight: 600;
    }
    .chip-value { 
        font-size: 0.82rem; 
        font-weight: 600; 
        color: #e2e8f0; 
    }

    /* Custom Alerts */
    .custom-alert { 
        border-radius: 9px; 
        padding: 12px 16px; 
        font-size: 0.88rem; 
        margin-bottom: 1rem; 
        display: flex; 
        align-items: flex-start; 
        gap: 10px; 
        border-left: 3px solid; 
    }
    .alert-error { 
        background: rgba(239, 68, 68, 0.1); 
        border-color: #ef4444; 
        color: #fca5a5; 
    }
    .alert-info { 
        background: rgba(59, 130, 246, 0.1); 
        border-color: #3b82f6; 
        color: #93c5fd; 
    }
    .alert-warning { 
        background: rgba(245, 158, 11, 0.1); 
        border-color: #f59e0b; 
        color: #fcd34d; 
    }

    /* Steps */
    .process-step { 
        display: flex; 
        align-items: flex-start; 
        gap: 14px; 
        padding: 12px 0; 
        border-bottom: 1px solid #1e293b; 
    }
    .process-step:last-child { 
        border-bottom: none; 
    }
    .step-num { 
        width: 28px; 
        height: 28px; 
        border-radius: 50%; 
        background: #1e3a8a; 
        color: #93c5fd; 
        font-size: 0.75rem; 
        font-weight: 700; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        flex-shrink: 0; 
        border: 1px solid #3b82f6;
    }
    .step-title { 
        font-size: 0.88rem; 
        font-weight: 600; 
        color: #f1f5f9; 
        margin-bottom: 2px; 
    }
    .step-desc { 
        font-size: 0.82rem; 
        color: #94a3b8; 
        line-height: 1.55; 
    }

    .disclaimer-box { 
        background: rgba(30, 41, 59, 0.5); 
        border: 1px solid #334155; 
        border-radius: 8px; 
        padding: 12px 14px; 
        font-size: 0.8rem; 
        color: #94a3b8; 
        line-height: 1.6; 
        margin-top: 1rem; 
    }
    
    .char-counter { 
        font-size: 0.78rem; 
        color: #64748b; 
        text-align: right; 
        margin-top: -6px; 
        margin-bottom: 10px; 
        font-weight: 500; 
        font-variant-numeric: tabular-nums;
    }

    @media (max-width: 768px) { 
        .hero-title { font-size: 1.6rem; } 
        .hero-container { padding: 1.75rem; } 
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_resources():
    """Load model, tokenizer, and slang dict once per app lifecycle."""
    from src.inference import load_model, load_tokenizer, load_slang_dict, get_device
    device = get_device()
    try:
        slang_dict = load_slang_dict(str(SLANG_PATH))
    except Exception as e:
        raise RuntimeError(f"Gagal memuat slang dictionary: {e}")
    try:
        tokenizer = load_tokenizer(str(MODEL_DIR))
    except Exception as e:
        raise RuntimeError(f"Gagal memuat tokenizer: {e}")
    try:
        model = load_model(str(MODEL_DIR), device)
    except Exception as e:
        raise RuntimeError(f"Gagal memuat model: {e}")
    return model, tokenizer, slang_dict, device


def render_top_navigation():
    """Render horizontal pill navigation at the top like reference design."""
    st.markdown('<div class="nav-label">Navigasi</div>', unsafe_allow_html=True)
    pages = ["Analyzer", "Cara Kerja", "Batch Analysis"]
    current_idx = pages.index(st.session_state.page) if st.session_state.page in pages else 0
    selected = st.radio(
        label="Navigasi Halaman",
        options=pages,
        index=current_idx,
        horizontal=True,
        label_visibility="collapsed",
        key="top_nav_radio",
    )
    if selected != st.session_state.page:
        st.session_state.page = selected
        st.rerun()


def render_sidebar(active_page: str):
    with st.sidebar:
        sidebar_header = (
            '<div class="sidebar-logo">'
            '<div class="sidebar-logo-text">MBG Analyzer</div>'
            '<div class="sidebar-logo-sub">Sentiment Analysis Platform</div>'
            '</div>'
            '<div class="sidebar-section-title">Navigasi</div>'
        )
        st.markdown(sidebar_header, unsafe_allow_html=True)

        pages = ["Analyzer", "Cara Kerja", "Batch Analysis"]
        for label in pages:
            btn_type = "primary" if label == active_page else "secondary"
            if st.button(label, key=f"nav_{label}", use_container_width=True, type=btn_type):
                st.session_state.page = label
                st.rerun()

        st.markdown('<div class="sidebar-section-title">Informasi Model</div>', unsafe_allow_html=True)
        chips = [
            ("Task", "Sentiment Classification"),
            ("Bahasa", "Bahasa Indonesia"),
            ("Model", "IndoBERT-base-p1"),
            ("Kelas", "Positive · Neutral · Negative"),
            ("Max Tokens", "256"),
            ("Framework", "PyTorch + Transformers"),
        ]
        for label, value in chips:
            chip_html = (
                f'<div class="sidebar-model-chip">'
                f'<div class="chip-label">{label}</div>'
                f'<div class="chip-value">{value}</div>'
                f'</div>'
            )
            st.markdown(chip_html, unsafe_allow_html=True)

        if st.session_state.get("history"):
            st.markdown('<div class="sidebar-section-title">Riwayat Analisis</div>', unsafe_allow_html=True)
            history_colors = {"Positive": "#22c55e", "Neutral": "#38bdf8", "Negative": "#f87171"}
            history_bgs = {
                "Positive": "rgba(34, 197, 94, 0.15)",
                "Neutral": "rgba(56, 189, 248, 0.15)",
                "Negative": "rgba(248, 113, 113, 0.15)",
            }
            for item in reversed(st.session_state.history[-5:]):
                color = history_colors.get(item["label"], "#94a3b8")
                bg = history_bgs.get(item["label"], "rgba(148, 163, 184, 0.15)")
                short = item["text"][:38] + "..." if len(item["text"]) > 38 else item["text"]
                item_html = (
                    f'<div class="history-item">'
                    f'<div class="history-text">{short}</div>'
                    f'<div class="history-badge" style="background:{bg};color:{color};">{item["label"]} {item["conf"]:.0%}</div>'
                    f'</div>'
                )
                st.markdown(item_html, unsafe_allow_html=True)


def render_hero():
    hero_html = (
        '<div class="hero-container">'
        '<div class="hero-badge">Analisis Sentimen Publik</div>'
        '<h1 class="hero-title">MBG <span>Sentiment</span> Analyzer</h1>'
        '<p class="hero-subtitle">'
        'Memahami sentimen komentar masyarakat terhadap program '
        '<strong style="color:#e2e8f0;">Makan Bergizi Gratis (MBG)</strong> &mdash; '
        'satu komentar dalam satu waktu.'
        '</p>'
        '<div class="hero-stats">'
        '<div class="hero-stat"><span class="hero-stat-value">3</span><span class="hero-stat-label">Kelas Sentimen</span></div>'
        '<div class="hero-stat"><span class="hero-stat-value">IndoBERT</span><span class="hero-stat-label">Model NLP</span></div>'
        '<div class="hero-stat"><span class="hero-stat-value">256</span><span class="hero-stat-label">Max Token</span></div>'
        '</div>'
        '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)


def select_sample_comment(text: str):
    """Callback to properly populate text area state and clear old result."""
    st.session_state["text_input_area"] = text
    st.session_state.pop("result", None)


def clear_text_input():
    """Callback to clear text area and result state."""
    st.session_state["text_input_area"] = ""
    st.session_state.pop("result", None)


def render_sample_comments():
    st.markdown('<div class="section-title">Coba dengan Contoh Komentar</div>', unsafe_allow_html=True)
    cols = st.columns(len(SAMPLE_COMMENTS))
    for i, (col, sample) in enumerate(zip(cols, SAMPLE_COMMENTS)):
        with col:
            st.button(
                sample["category"],
                key=f"sample_{i}",
                use_container_width=True,
                help=sample["text"],
                on_click=select_sample_comment,
                args=(sample["text"],),
            )


def render_probability_bars(probs_dict: Dict[str, float]):
    bar_config = {
        "Positive": {"color": "#22c55e", "label": "Positif"},
        "Neutral": {"color": "#38bdf8", "label": "Netral"},
        "Negative": {"color": "#f87171", "label": "Negatif"},
    }
    rows = []
    for label, prob in probs_dict.items():
        cfg = bar_config[label]
        pct = prob * 100
        rows.append(
            f'<div class="prob-row">'
            f'<div class="prob-label"><span class="prob-dot" style="background:{cfg["color"]};"></span>{cfg["label"]}</div>'
            f'<div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:{pct:.1f}%;background:{cfg["color"]};"></div></div>'
            f'<div class="prob-value">{pct:.1f}%</div>'
            f'</div>'
        )
    bars_content = "".join(rows)
    prob_html = (
        f'<div class="prob-section">'
        f'<div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;margin-bottom:12px;">Distribusi Probabilitas</div>'
        f'{bars_content}'
        f'</div>'
    )
    st.markdown(prob_html, unsafe_allow_html=True)


def render_result_card(label: str, confidence: float, probs_dict: Dict[str, float]):
    cfg = SENTIMENT_CONFIG[label]
    conf_pct = f"{confidence * 100:.1f}%"
    p_pos = probs_dict["Positive"] * 100
    p_neu = probs_dict["Neutral"] * 100
    p_neg = probs_dict["Negative"] * 100

    card_html = (
        f'<div class="result-container" style="background:{cfg["bg_color"]};border-color:{cfg["border_color"]};">'
        f'<div class="result-header">'
        f'<div>'
        f'<div class="result-badge" style="background:{cfg["badge_bg"]};color:{cfg["color"]};">{cfg["label_id"]}</div>'
        f'<div class="result-label" style="color:{cfg["color"]};">{label}</div>'
        f'</div>'
        f'</div>'
        f'<div class="confidence-block">'
        f'<div class="confidence-label">Confidence Score</div>'
        f'<div class="confidence-value" style="color:{cfg["color"]};">{conf_pct}</div>'
        f'</div>'
        f'<div class="result-description">{cfg["description"]}</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    render_probability_bars(probs_dict)

    interp_html = (
        f'<div style="margin-top:1.25rem;padding:14px 16px;background:#0f172a;border-radius:10px;border:1px solid #1e293b;">'
        f'<div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;margin-bottom:6px;">Interpretasi</div>'
        f'<div style="font-size:0.88rem;color:#cbd5e1;line-height:1.65;">'
        f'Model memprediksi komentar ini memiliki sentimen '
        f'<strong style="color:{cfg["color"]};">{label}</strong> '
        f'dengan confidence sebesar <strong>{conf_pct}</strong>.<br>'
        f'Detail Probabilitas: Positif {p_pos:.1f}% &nbsp;|&nbsp; Netral {p_neu:.1f}% &nbsp;|&nbsp; Negatif {p_neg:.1f}%'
        f'</div>'
        f'</div>'
        f'<div class="disclaimer-box">'
        f'<strong>Catatan:</strong> Hasil merupakan prediksi model Machine Learning dan dapat mengandung ketidakakuratan pada komentar ambigu, sarkastik, atau bernada kontekstual.'
        f'</div>'
    )
    st.markdown(interp_html, unsafe_allow_html=True)


def page_analyzer(model, tokenizer, slang_dict, device):
    from src.inference import predict_sentiment, validate_input

    render_hero()

    # Sample comments section wrapped in border container
    with st.container(border=True):
        render_sample_comments()

    # Input section wrapped in border container
    with st.container(border=True):
        st.markdown('<div class="section-title">Masukkan Komentar untuk Dianalisis</div>', unsafe_allow_html=True)

        input_text = st.text_area(
            label="Komentar MBG",
            placeholder="Tulis komentar di sini... (atau pilih salah satu contoh di atas)",
            height=130,
            max_chars=MAX_INPUT_CHARS,
            label_visibility="collapsed",
            key="text_input_area",
        )

        current_len = len(input_text)
        st.markdown(
            f'<div class="char-counter"><span id="live-char-count">{current_len}</span> / {MAX_INPUT_CHARS:,} karakter</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_btn = st.button("Analisis Sentimen", key="analyze_btn", use_container_width=True, type="primary")
        with col2:
            st.button("Bersihkan", key="clear_btn", use_container_width=True, on_click=clear_text_input, type="secondary")

    # Real-time character counter script that continuously synchronizes counter with textarea.value
    components.html("""
    <script>
    (function() {
        function syncLiveCounter() {
            try {
                const doc = window.parent.document;
                const textarea = doc.querySelector('textarea');
                const counter = doc.getElementById('live-char-count');
                if (textarea && counter) {
                    const currentLen = textarea.value.length;
                    if (counter.textContent !== String(currentLen)) {
                        counter.textContent = currentLen;
                    }
                    if (!textarea.dataset.hasLiveCounter) {
                        textarea.dataset.hasLiveCounter = "true";
                        ['input', 'keyup', 'paste', 'change'].forEach(evt => {
                            textarea.addEventListener(evt, () => {
                                counter.textContent = textarea.value.length;
                            });
                        });
                    }
                }
            } catch(e) {}
        }
        syncLiveCounter();
        setInterval(syncLiveCounter, 100);
    })();
    </script>
    """, height=0, width=0)

    # Inference execution
    if analyze_btn:
        active_text = st.session_state.get("text_input_area", "").strip()
        is_valid, error_msg = validate_input(active_text)
        if not is_valid:
            st.markdown(
                f'<div class="custom-alert alert-error"><span>{error_msg}</span></div>',
                unsafe_allow_html=True,
            )
            return

        with st.spinner("Menganalisis sentimen komentar..."):
            try:
                t0 = time.time()
                label, confidence, probs_dict = predict_sentiment(
                    raw_text=active_text,
                    tokenizer=tokenizer,
                    model=model,
                    slang_dict=slang_dict,
                    device=device,
                )
                logger.info(f"Prediction: {label} ({confidence:.3f}) | {time.time()-t0:.2f}s")
            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                st.markdown("""
                <div class="custom-alert alert-error">
                    <span><strong>Analisis gagal.</strong> Terjadi kesalahan saat memproses teks. Silakan coba komentar lain.</span>
                </div>""", unsafe_allow_html=True)
                return

        st.session_state.result = {
            "label": label, "confidence": confidence,
            "probs_dict": probs_dict, "text": active_text,
        }

        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({"text": active_text, "label": label, "conf": confidence})
        if len(st.session_state.history) > 10:
            st.session_state.history = st.session_state.history[-10:]

    # Result display
    if st.session_state.get("result"):
        res = st.session_state.result
        with st.container(border=True):
            st.markdown('<div class="section-title">Hasil Analisis</div>', unsafe_allow_html=True)
            render_result_card(label=res["label"], confidence=res["confidence"], probs_dict=res["probs_dict"])


def page_cara_kerja():
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="section-title">Cara Kerja Sistem</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:0.9rem;color:#cbd5e1;line-height:1.7;margin-bottom:1.5rem;">
            MBG Sentiment Analyzer menggunakan pipeline NLP yang dirancang khusus untuk teks media sosial
            Bahasa Indonesia, termasuk normalisasi slang, penanganan kata informal, dan klasifikasi transformer.
        </p>
        """, unsafe_allow_html=True)

        steps = [
            ("Input Teks", "Pengguna memasukkan komentar dalam Bahasa Indonesia, termasuk slang, singkatan, atau gaya bahasa informal media sosial."),
            ("Unicode Normalization", "Teks dinormalisasi menggunakan standar NFKC untuk konsistensi encoding karakter."),
            ("Penggantian Emoji", "Emoji dikonversi ke representasi semantik Bahasa Indonesia agar dapat dipahami model NLP."),
            ("Text Cleaning", "Pembersihan URL, mention, hashtag, dan konversi ke huruf kecil."),
            ("Normalisasi Slang", 'Kata slang/singkatan dinormalisasi menggunakan kamus khusus (contoh: "gak" → "tidak", "bgt" → "banget").'),
            ("Tokenisasi", "Teks diubah menjadi token oleh BertTokenizer (max 256 token, padding, truncation)."),
            ("Model Inference", "Token diproses oleh IndoBERT-base-p1 fine-tuned yang menghasilkan logits untuk 3 kelas sentimen."),
            ("Probabilitas", "Logits dikonversi ke probabilitas via fungsi Softmax."),
            ("Prediksi Akhir", "Kelas dengan probabilitas tertinggi dipilih: Positive, Neutral, atau Negative."),
        ]
        for i, (title, desc) in enumerate(steps, 1):
            step_html = (
                f'<div class="process-step">'
                f'<div class="step-num">{i}</div>'
                f'<div><div class="step-title">{title}</div><div class="step-desc">{desc}</div></div>'
                f'</div>'
            )
            st.markdown(step_html, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-title">Tentang Model</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:0.9rem;color:#cbd5e1;line-height:1.7;">
            Model ini menggunakan arsitektur <strong>IndoBERT-base-p1</strong> dengan pipeline 
            <strong>Stream B Preprocessing</strong> (normalisasi slang + tokenisasi khusus).
        </p>
        <ul style="font-size:0.88rem;color:#94a3b8;line-height:1.9;padding-left:1.2rem;">
            <li><strong>Model:</strong> IndoBERT-base-p1 (Fine-tuned Sequence Classification)</li>
            <li><strong>Max Length:</strong> 256 tokens</li>
            <li><strong>Validation Performance:</strong> OOF Macro F1 ~0.8997</li>
        </ul>
        <div class="disclaimer-box">
            <strong>Transparansi:</strong> Prediksi model merupakan hasil inferensi machine learning dan dapat mengandung bias atau ketidakakuratan pada teks bernada sarkasme atau konteks sosial khusus.
        </div>
        """, unsafe_allow_html=True)


def page_batch_analysis(model, tokenizer, slang_dict, device):
    from src.inference import predict_sentiment, validate_input
    import pandas as pd

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="section-title">Batch Analysis &mdash; Upload CSV</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:0.88rem;color:#cbd5e1;line-height:1.65;margin-bottom:1rem;">
            Upload file CSV dengan kolom <code>comment</code> untuk menganalisis banyak komentar sekaligus.
            Hasil dapat diunduh dalam format CSV.
        </p>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload CSV (kolom: comment)",
            type=["csv"],
            key="batch_upload",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.markdown(
                    f'<div class="custom-alert alert-error"><span>Gagal membaca file: {e}</span></div>',
                    unsafe_allow_html=True,
                )
                return

            if "comment" not in df.columns:
                st.markdown(
                    '<div class="custom-alert alert-error"><span>File CSV harus memiliki kolom <code>comment</code>.</span></div>',
                    unsafe_allow_html=True,
                )
                return

            df["comment"] = df["comment"].fillna("").astype(str)
            total = len(df)

            if total > 500:
                st.markdown(
                    '<div class="custom-alert alert-warning"><span>File memiliki lebih dari 500 baris. Proses mungkin membutuhkan beberapa menit.</span></div>',
                    unsafe_allow_html=True,
                )

            results = []
            progress = st.progress(0, text="Memproses komentar...")

            for i, row in df.iterrows():
                text = str(row["comment"])
                is_valid, _ = validate_input(text)
                if not is_valid:
                    results.append({
                        "comment": text, "sentiment": "Error", "confidence": 0.0,
                        "positive_probability": 0.0, "neutral_probability": 0.0, "negative_probability": 0.0,
                    })
                else:
                    try:
                        label, confidence, probs_dict = predict_sentiment(text, tokenizer, model, slang_dict, device)
                        results.append({
                            "comment": text, "sentiment": label,
                            "confidence": round(confidence, 4),
                            "positive_probability": round(probs_dict["Positive"], 4),
                            "neutral_probability": round(probs_dict["Neutral"], 4),
                            "negative_probability": round(probs_dict["Negative"], 4),
                        })
                    except Exception as e:
                        logger.error(f"Batch error row {i}: {e}")
                        results.append({
                            "comment": text, "sentiment": "Error", "confidence": 0.0,
                            "positive_probability": 0.0, "neutral_probability": 0.0, "negative_probability": 0.0,
                        })
                progress.progress((i + 1) / total, text=f"Memproses {i+1}/{total}...")

            progress.empty()
            result_df = pd.DataFrame(results)
            dist = result_df[result_df["sentiment"] != "Error"]["sentiment"].value_counts()

            summary_chips = (
                f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin:1rem 0;">'
                f'<div class="sidebar-model-chip" style="flex:1;background:rgba(34,197,94,0.1);border-color:rgba(34,197,94,0.3);">'
                f'<div class="chip-label">Positive</div>'
                f'<div class="chip-value" style="color:#22c55e;font-size:1.4rem;">{dist.get("Positive", 0):,}</div>'
                f'</div>'
                f'<div class="sidebar-model-chip" style="flex:1;background:rgba(56,189,248,0.1);border-color:rgba(56,189,248,0.3);">'
                f'<div class="chip-label">Neutral</div>'
                f'<div class="chip-value" style="color:#38bdf8;font-size:1.4rem;">{dist.get("Neutral", 0):,}</div>'
                f'</div>'
                f'<div class="sidebar-model-chip" style="flex:1;background:rgba(248,113,113,0.1);border-color:rgba(248,113,113,0.3);">'
                f'<div class="chip-label">Negative</div>'
                f'<div class="chip-value" style="color:#f87171;font-size:1.4rem;">{dist.get("Negative", 0):,}</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(summary_chips, unsafe_allow_html=True)

            st.dataframe(result_df, use_container_width=True, height=300)
            csv_data = result_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "Download Hasil CSV",
                csv_data,
                "mbg_sentiment_results.csv",
                "text/csv",
                use_container_width=True,
            )


def main():
    inject_css()

    # Initialize session state
    for key, default in [("page", "Analyzer"), ("history", []), ("result", None), ("text_input_area", "")]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Load model resources (cached)
    try:
        with st.spinner("Memuat model..."):
            model, tokenizer, slang_dict, device = load_resources()
    except RuntimeError as e:
        st.markdown(
            f'<div class="custom-alert alert-error"><span><strong>Gagal memuat model.</strong><br>{e}</span></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    render_sidebar(st.session_state.page)

    # Top horizontal navigation (per-page capsule like reference image)
    render_top_navigation()

    page = st.session_state.page
    if page == "Analyzer":
        page_analyzer(model, tokenizer, slang_dict, device)
    elif page == "Cara Kerja":
        page_cara_kerja()
    elif page == "Batch Analysis":
        page_batch_analysis(model, tokenizer, slang_dict, device)


if __name__ == "__main__":
    main()
