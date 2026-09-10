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
        "icon": "✅",
        "color": "#16a34a",
        "bg_color": "#f0fdf4",
        "border_color": "#86efac",
        "badge_bg": "#dcfce7",
        "label_id": "SENTIMEN POSITIF",
        "description": "Komentar ini diprediksi mengandung sentimen <strong>positif</strong> &mdash; menunjukkan dukungan, apresiasi, atau pandangan optimis terhadap program MBG.",
    },
    "Neutral": {
        "icon": "ℹ️",
        "color": "#2563eb",
        "bg_color": "#eff6ff",
        "border_color": "#93c5fd",
        "badge_bg": "#dbeafe",
        "label_id": "SENTIMEN NETRAL",
        "description": "Komentar ini diprediksi bersifat <strong>netral</strong> &mdash; menyampaikan informasi atau pandangan seimbang tanpa kecenderungan positif maupun negatif yang kuat.",
    },
    "Negative": {
        "icon": "⚠️",
        "color": "#dc2626",
        "bg_color": "#fef2f2",
        "border_color": "#fca5a5",
        "badge_bg": "#fee2e2",
        "label_id": "SENTIMEN NEGATIF",
        "description": "Komentar ini diprediksi mengandung sentimen <strong>negatif</strong> &mdash; menunjukkan kritik, kekecewaan, atau pandangan yang tidak mendukung program MBG.",
    },
}

SAMPLE_COMMENTS = [
    {
        "category": "Positif",
        "icon": "✅",
        "text": "Alhamdulillah program MBG sangat bermanfaat dan membantu gizi anak sekolah di pelosok. Menunya bergizi, sehat, dan anak-anak senang banget! Terima kasih pemerintah 👍",
    },
    {
        "category": "Netral",
        "icon": "ℹ️",
        "text": "Pemerintah menjadwalkan uji coba distribusi program makan bergizi gratis di 50 sekolah wilayah Jawa Barat mulai pekan depan.",
    },
    {
        "category": "Negatif",
        "icon": "⚠️",
        "text": "Program MBG ini parah banget pelaksanaannya, makanannya basi dan anggarannya malah dikorupsi oknum! Sangat mengecewakan dan merugikan rakyat kecil.",
    },
    {
        "category": "Ambigu/Sarkasme",
        "icon": "🗿",
        "text": "Katanya program MBG anggarannya triliunan buat rakyat kecil, tapi kok lauknya cuma tempe seiprit doang? Mantap bgt deh solusinya.",
    },
]

st.set_page_config(
    page_title="MBG Sentiment Analyzer",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "MBG Sentiment Analyzer — Analisis sentimen komentar masyarakat terhadap program Makan Bergizi Gratis."},
)


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
    .stApp { background-color: #f8fafc; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; max-width: 900px !important; }

    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f3460 100%);
        border-radius: 20px; padding: 3rem; margin-bottom: 2rem;
        position: relative; overflow: hidden;
        box-shadow: 0 20px 60px rgba(15,23,42,0.25);
    }
    .hero-container::before {
        content: ''; position: absolute; top: -40%; right: -10%; width: 60%; height: 200%;
        background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(59,130,246,0.2); border: 1px solid rgba(59,130,246,0.4);
        color: #93c5fd; font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.08em; padding: 4px 12px; border-radius: 100px;
        margin-bottom: 1rem; text-transform: uppercase;
    }
    .hero-title { font-size: 2.4rem; font-weight: 800; color: #ffffff; margin: 0 0 0.5rem 0; line-height: 1.15; letter-spacing: -0.02em; }
    .hero-title span { background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .hero-subtitle { font-size: 1rem; color: #94a3b8; margin: 0 0 1.8rem 0; line-height: 1.6; max-width: 540px; }
    .hero-stats { display: flex; gap: 2rem; flex-wrap: wrap; }
    .hero-stat { display: flex; flex-direction: column; gap: 2px; }
    .hero-stat-value { font-size: 1.4rem; font-weight: 700; color: #ffffff; }
    .hero-stat-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }

    .section-card { background: #ffffff; border-radius: 16px; padding: 1.75rem; margin-bottom: 1.5rem; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03); }
    .section-title { font-size: 0.85rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem; }

    .stTextArea textarea { font-family: 'Inter', sans-serif !important; font-size: 0.95rem !important; color: #1e293b !important; border-radius: 12px !important; border: 1.5px solid #e2e8f0 !important; background: #f8fafc !important; padding: 14px !important; line-height: 1.65 !important; transition: border-color 0.2s ease, box-shadow 0.2s ease !important; }
    .stTextArea textarea:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important; outline: none !important; background: #ffffff !important; }

    .stButton > button { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; font-size: 0.95rem !important; border-radius: 10px !important; padding: 0.65rem 1.75rem !important; transition: all 0.2s ease !important; }

    .result-container { border-radius: 16px; padding: 2rem; border: 1.5px solid; animation: slideInUp 0.4s ease; }
    @keyframes slideInUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
    .result-header { display: flex; align-items: center; gap: 14px; margin-bottom: 1.25rem; }
    .result-icon { width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; flex-shrink: 0; }
    .result-badge { display: inline-flex; align-items: center; gap: 6px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; padding: 5px 12px; border-radius: 100px; text-transform: uppercase; }
    .result-label { font-size: 1.65rem; font-weight: 800; letter-spacing: -0.01em; margin: 4px 0 0 0; }
    .confidence-block { background: rgba(255,255,255,0.7); border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.8); }
    .confidence-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 6px; }
    .confidence-value { font-size: 2.2rem; font-weight: 800; line-height: 1; letter-spacing: -0.02em; }
    .result-description { font-size: 0.88rem; line-height: 1.65; color: #475569; margin-top: 0.75rem; }

    .prob-section { margin-top: 1.5rem; }
    .prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .prob-label { font-size: 0.82rem; font-weight: 600; color: #374151; min-width: 72px; }
    .prob-bar-wrap { flex: 1; height: 8px; background: #e5e7eb; border-radius: 100px; overflow: hidden; }
    .prob-bar-fill { height: 100%; border-radius: 100px; }
    .prob-value { font-size: 0.82rem; font-weight: 700; color: #1e293b; min-width: 46px; text-align: right; }

    .history-item { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; border-radius: 10px; background: #f8fafc; border: 1px solid #e2e8f0; margin-bottom: 8px; }
    .history-text { font-size: 0.82rem; color: #475569; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 170px; }
    .history-badge { font-size: 0.72rem; font-weight: 600; padding: 3px 10px; border-radius: 100px; white-space: nowrap; }

    [data-testid="stSidebar"] { background: #0f172a !important; }
    [data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
    .sidebar-logo { display: flex; align-items: center; gap: 10px; padding: 0.5rem 0 1.5rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 1.5rem; }
    .sidebar-logo-text { font-size: 1rem; font-weight: 700; color: #f8fafc; }
    .sidebar-logo-sub { font-size: 0.72rem; color: #64748b; }
    .sidebar-section-title { font-size: 0.7rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; margin: 1.5rem 0 0.75rem 0; }
    .sidebar-model-chip { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; }
    .chip-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 3px; }
    .chip-value { font-size: 0.85rem; font-weight: 600; color: #e2e8f0; }

    .custom-alert { border-radius: 10px; padding: 12px 16px; font-size: 0.88rem; margin-bottom: 1rem; display: flex; align-items: flex-start; gap: 10px; border-left: 3px solid; }
    .alert-error { background: #fef2f2; border-color: #ef4444; color: #991b1b; }
    .alert-info { background: #eff6ff; border-color: #3b82f6; color: #1e40af; }
    .alert-warning { background: #fffbeb; border-color: #f59e0b; color: #92400e; }

    .process-step { display: flex; align-items: flex-start; gap: 14px; padding: 14px 0; border-bottom: 1px solid #f1f5f9; }
    .process-step:last-child { border-bottom: none; }
    .step-num { width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #2563eb, #7c3aed); color: white; font-size: 0.78rem; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
    .step-title { font-size: 0.9rem; font-weight: 600; color: #1e293b; margin-bottom: 3px; }
    .step-desc { font-size: 0.82rem; color: #64748b; line-height: 1.55; }

    .disclaimer-box { background: #fefce8; border: 1px solid #fde68a; border-radius: 10px; padding: 14px 16px; font-size: 0.83rem; color: #713f12; line-height: 1.6; display: flex; gap: 10px; align-items: flex-start; margin-top: 1rem; }
    .char-counter { font-size: 0.78rem; color: #94a3b8; text-align: right; margin-top: -8px; margin-bottom: 8px; font-weight: 500; }
    .char-counter.near-limit { color: #f59e0b; }
    .char-counter.at-limit { color: #ef4444; }

    @media (max-width: 768px) { .hero-title { font-size: 1.7rem; } .hero-container { padding: 2rem; } }
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


def render_sidebar(active_page: str):
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div style="font-size:1.8rem">🍱</div>
            <div>
                <div class="sidebar-logo-text">MBG Analyzer</div>
                <div class="sidebar-logo-sub">Sentiment Analysis Tool</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-title">Navigasi</div>', unsafe_allow_html=True)
        pages = [("🔍", "Analyzer"), ("ℹ️", "Cara Kerja"), ("📦", "Batch Analysis")]
        for icon, label in pages:
            if st.sidebar.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
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
            st.markdown(f"""
            <div class="sidebar-model-chip">
                <div class="chip-label">{label}</div>
                <div class="chip-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.get("history"):
            st.markdown('<div class="sidebar-section-title">Riwayat Analisis</div>', unsafe_allow_html=True)
            history_colors = {"Positive": "#16a34a", "Neutral": "#2563eb", "Negative": "#dc2626"}
            history_bgs = {"Positive": "#dcfce7", "Neutral": "#dbeafe", "Negative": "#fee2e2"}
            for item in reversed(st.session_state.history[-5:]):
                color = history_colors.get(item["label"], "#64748b")
                bg = history_bgs.get(item["label"], "#f1f5f9")
                short = item["text"][:40] + "..." if len(item["text"]) > 40 else item["text"]
                st.markdown(f"""
                <div class="history-item">
                    <div class="history-text">{short}</div>
                    <div class="history-badge" style="background:{bg};color:{color};">{item["label"]} {item["conf"]:.0%}</div>
                </div>
                """, unsafe_allow_html=True)


def render_hero():
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">🇮🇩 &nbsp; Analisis Sentimen Publik</div>
        <h1 class="hero-title">MBG <span>Sentiment</span> Analyzer</h1>
        <p class="hero-subtitle">
            Memahami sentimen komentar masyarakat terhadap program
            <strong style="color:#e2e8f0;">Makan Bergizi Gratis (MBG)</strong> &mdash;
            satu komentar dalam satu waktu.
        </p>
        <div class="hero-stats">
            <div class="hero-stat"><span class="hero-stat-value">3</span><span class="hero-stat-label">Kelas Sentimen</span></div>
            <div class="hero-stat"><span class="hero-stat-value">IndoBERT</span><span class="hero-stat-label">Model NLP</span></div>
            <div class="hero-stat"><span class="hero-stat-value">256</span><span class="hero-stat-label">Max Token</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sample_comments():
    st.markdown('<div class="section-title">💬 Coba dengan Contoh Komentar</div>', unsafe_allow_html=True)
    cols = st.columns(len(SAMPLE_COMMENTS))
    for i, (col, sample) in enumerate(zip(cols, SAMPLE_COMMENTS)):
        with col:
            if st.button(
                f"{sample['icon']} {sample['category']}",
                key=f"sample_{i}",
                use_container_width=True,
                help=sample["text"],
            ):
                st.session_state.input_text = sample["text"]
                st.session_state.pop("result", None)
                st.rerun()


def render_probability_bars(probs_dict: Dict[str, float]):
    bar_config = {
        "Positive": {"color": "#16a34a", "icon": "✅"},
        "Neutral": {"color": "#2563eb", "icon": "ℹ️"},
        "Negative": {"color": "#dc2626", "icon": "⚠️"},
    }
    bars_html = ""
    for label, prob in probs_dict.items():
        cfg = bar_config[label]
        pct = prob * 100
        bars_html += f"""
        <div class="prob-row">
            <div class="prob-label">{cfg['icon']} {label}</div>
            <div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:{pct:.1f}%;background:{cfg['color']};"></div></div>
            <div class="prob-value">{pct:.1f}%</div>
        </div>"""
    st.markdown(f"""
    <div class="prob-section">
        <div style="font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;margin-bottom:12px;">Distribusi Probabilitas</div>
        {bars_html}
    </div>""", unsafe_allow_html=True)


def render_result_card(label: str, confidence: float, probs_dict: Dict[str, float]):
    cfg = SENTIMENT_CONFIG[label]
    conf_pct = f"{confidence * 100:.1f}%"
    p_pos = probs_dict["Positive"] * 100
    p_neu = probs_dict["Neutral"] * 100
    p_neg = probs_dict["Negative"] * 100

    st.markdown(f"""
    <div class="result-container" style="background:{cfg['bg_color']};border-color:{cfg['border_color']};">
        <div class="result-header">
            <div class="result-icon" style="background:{cfg['badge_bg']};">{cfg['icon']}</div>
            <div>
                <div class="result-badge" style="background:{cfg['badge_bg']};color:{cfg['color']};">{cfg['label_id']}</div>
                <div class="result-label" style="color:{cfg['color']};">{label}</div>
            </div>
        </div>
        <div class="confidence-block">
            <div class="confidence-label">Confidence Score</div>
            <div class="confidence-value" style="color:{cfg['color']};">{conf_pct}</div>
        </div>
        <div class="result-description">{cfg['description']}</div>
    </div>""", unsafe_allow_html=True)

    render_probability_bars(probs_dict)

    st.markdown(f"""
    <div style="margin-top:1.25rem;padding:14px 16px;background:#f8fafc;border-radius:10px;border:1px solid #e2e8f0;">
        <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;margin-bottom:8px;">Interpretasi</div>
        <div style="font-size:0.88rem;color:#374151;line-height:1.65;">
            Model memprediksi komentar ini memiliki sentimen
            <strong style="color:{cfg['color']};">{label}</strong>
            dengan confidence sebesar <strong>{conf_pct}</strong>.<br>
            Probabilitas: Positive {p_pos:.1f}% &nbsp;|&nbsp; Neutral {p_neu:.1f}% &nbsp;|&nbsp; Negative {p_neg:.1f}%
        </div>
    </div>
    <div class="disclaimer-box">
        <span>⚠️</span>
        <span><strong>Catatan:</strong> Hasil merupakan prediksi model Machine Learning dan dapat mengandung ketidakakuratan, terutama pada komentar ambigu, sarkastik, atau sangat informal.</span>
    </div>""", unsafe_allow_html=True)


def page_analyzer(model, tokenizer, slang_dict, device):
    from src.inference import predict_sentiment, validate_input

    render_hero()

    # Sample comments section
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_sample_comments()
        st.markdown('</div>', unsafe_allow_html=True)

    # Input section
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">✍️ Masukkan Komentar untuk Dianalisis</div>', unsafe_allow_html=True)

        input_text = st.text_area(
            label="Komentar MBG",
            value=st.session_state.get("input_text", ""),
            placeholder='Contoh: "Program MBG sangat membantu anak-anak mendapatkan makanan bergizi setiap hari."',
            height=140,
            max_chars=MAX_INPUT_CHARS,
            label_visibility="collapsed",
            key="text_input_area",
        )

        char_count = len(input_text)
        counter_class = "at-limit" if char_count > MAX_INPUT_CHARS * 0.9 else (
            "near-limit" if char_count > MAX_INPUT_CHARS * 0.75 else ""
        )
        st.markdown(
            f'<div class="char-counter {counter_class}">{char_count:,} / {MAX_INPUT_CHARS:,} karakter</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_btn = st.button("🔍  Analisis Sentimen", key="analyze_btn", use_container_width=True)
        with col2:
            clear_btn = st.button("🗑️  Bersihkan", key="clear_btn", use_container_width=True)

        if clear_btn:
            st.session_state.input_text = ""
            st.session_state.pop("result", None)
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # Inference
    if analyze_btn:
        is_valid, error_msg = validate_input(input_text)
        if not is_valid:
            st.markdown(
                f'<div class="custom-alert alert-error"><span>❌</span><span>{error_msg}</span></div>',
                unsafe_allow_html=True,
            )
            return

        with st.spinner("Menganalisis sentimen komentar..."):
            try:
                t0 = time.time()
                label, confidence, probs_dict = predict_sentiment(
                    raw_text=input_text,
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
                    <span>❌</span>
                    <span><strong>Analisis gagal.</strong> Terjadi kesalahan saat memproses teks. Silakan coba komentar lain.</span>
                </div>""", unsafe_allow_html=True)
                return

        st.session_state.result = {
            "label": label, "confidence": confidence,
            "probs_dict": probs_dict, "text": input_text,
        }

        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({"text": input_text, "label": label, "conf": confidence})
        if len(st.session_state.history) > 10:
            st.session_state.history = st.session_state.history[-10:]

    # Result display
    if st.session_state.get("result"):
        res = st.session_state.result
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Hasil Analisis</div>', unsafe_allow_html=True)
        render_result_card(label=res["label"], confidence=res["confidence"], probs_dict=res["probs_dict"])
        st.markdown('</div>', unsafe_allow_html=True)


def page_cara_kerja():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-card">
        <div class="section-title">⚙️ Cara Kerja Sistem</div>
        <p style="font-size:0.9rem;color:#475569;line-height:1.7;margin-bottom:1.5rem;">
            MBG Sentiment Analyzer menggunakan pipeline NLP yang dirancang khusus untuk teks media sosial
            Bahasa Indonesia, termasuk slang, singkatan, emoji, dan gaya bahasa informal.
        </p>
    """, unsafe_allow_html=True)

    steps = [
        ("Input Teks", "Pengguna memasukkan komentar dalam Bahasa Indonesia, termasuk slang, emoji, singkatan, atau gaya bahasa informal media sosial."),
        ("Unicode Normalization", "Teks dinormalisasi menggunakan standar NFKC untuk konsistensi encoding karakter."),
        ("Penggantian Emoji", "Emoji dikonversi ke tag semantik Bahasa Indonesia (contoh: 😂 → emoji_tertawa) agar dapat diproses model NLP."),
        ("Text Cleaning", "URL, @mentions, #hashtags dihapus. Teks dikonversi ke huruf kecil."),
        ("Normalisasi Slang", 'Kata slang/singkatan dinormalisasi menggunakan kamus khusus (contoh: "gak" → "tidak", "bgt" → "banget").'),
        ("Tokenisasi", "Teks diubah menjadi token oleh BertTokenizer (max 256 token, padding, truncation)."),
        ("Model Inference", "Token diproses oleh IndoBERT-base-p1 fine-tuned yang menghasilkan logits untuk 3 kelas sentimen."),
        ("Probabilitas", "Logits dikonversi ke probabilitas via fungsi Softmax."),
        ("Prediksi Akhir", "Kelas dengan probabilitas tertinggi dipilih: Positive, Neutral, atau Negative."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="process-step">
            <div class="step-num">{i}</div>
            <div><div class="step-title">{title}</div><div class="step-desc">{desc}</div></div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">🤖 Tentang Model</div>
        <p style="font-size:0.9rem;color:#475569;line-height:1.7;">
            Model ini adalah bagian dari pipeline <strong>Dual-Stream Transformer Ensemble</strong>
            yang mengintegrasikan tiga komponen NLP:
        </p>
        <ul style="font-size:0.88rem;color:#475569;line-height:1.9;padding-left:1.2rem;">
            <li><strong>Model 1:</strong> Calibrated LinearSVC (TF-IDF + Meta-Features)</li>
            <li><strong>Model 2:</strong> IndoBERTweet-base (Stream A &mdash; teks informal Twitter)</li>
            <li><strong>Model 3:</strong> IndoBERT-base-p1 (Stream B &mdash; normalisasi slang)</li>
        </ul>
        <p style="font-size:0.88rem;color:#475569;line-height:1.7;">
            Aplikasi ini menggunakan <strong>Model 3 (IndoBERT-base-p1)</strong> dengan
            <strong>Stream B preprocessing</strong>. Model mencapai OOF Macro F1 sebesar
            <strong>0.8997</strong> dan ensemble final mencapai <strong>LB: 0.93395</strong>.
        </p>
        <div class="disclaimer-box">
            <span>ℹ️</span>
            <span><strong>Transparansi:</strong> Prediksi model bukan kebenaran absolut dan dapat mengandung kesalahan pada komentar ambigu, sarkastik, atau memiliki konteks sosial kompleks.</span>
        </div>
    </div>""", unsafe_allow_html=True)


def page_batch_analysis(model, tokenizer, slang_dict, device):
    from src.inference import predict_sentiment, validate_input
    import pandas as pd

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📦 Batch Analysis &mdash; Upload CSV</div>
        <p style="font-size:0.88rem;color:#475569;line-height:1.65;margin-bottom:1rem;">
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
                f'<div class="custom-alert alert-error"><span>❌</span><span>Gagal membaca file: {e}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
            return

        if "comment" not in df.columns:
            st.markdown(
                '<div class="custom-alert alert-error"><span>❌</span><span>File CSV harus memiliki kolom <code>comment</code>.</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
            return

        df["comment"] = df["comment"].fillna("").astype(str)
        total = len(df)

        if total > 500:
            st.markdown(
                '<div class="custom-alert alert-warning"><span>⚠️</span><span>File memiliki lebih dari 500 baris. Proses mungkin membutuhkan beberapa menit.</span></div>',
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

        st.markdown(f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin:1rem 0;">
            <div class="sidebar-model-chip" style="flex:1;background:#f0fdf4;border-color:#86efac;">
                <div class="chip-label">Positive</div>
                <div class="chip-value" style="color:#16a34a;font-size:1.4rem;">{dist.get("Positive", 0):,}</div>
            </div>
            <div class="sidebar-model-chip" style="flex:1;background:#eff6ff;border-color:#93c5fd;">
                <div class="chip-label">Neutral</div>
                <div class="chip-value" style="color:#2563eb;font-size:1.4rem;">{dist.get("Neutral", 0):,}</div>
            </div>
            <div class="sidebar-model-chip" style="flex:1;background:#fef2f2;border-color:#fca5a5;">
                <div class="chip-label">Negative</div>
                <div class="chip-value" style="color:#dc2626;font-size:1.4rem;">{dist.get("Negative", 0):,}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.dataframe(result_df, use_container_width=True, height=300)
        csv_data = result_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "⬇️  Download Hasil CSV",
            csv_data,
            "mbg_sentiment_results.csv",
            "text/csv",
            use_container_width=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


def main():
    inject_css()

    # Initialize session state
    for key, default in [("page", "Analyzer"), ("history", []), ("result", None), ("input_text", "")]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Load model resources (cached)
    try:
        with st.spinner("⏳ Memuat model..."):
            model, tokenizer, slang_dict, device = load_resources()
    except RuntimeError as e:
        st.markdown(
            f'<div class="custom-alert alert-error"><span>❌</span><span><strong>Gagal memuat model.</strong><br>{e}</span></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    render_sidebar(st.session_state.page)

    page = st.session_state.page
    if page == "Analyzer":
        page_analyzer(model, tokenizer, slang_dict, device)
    elif page == "Cara Kerja":
        page_cara_kerja()
    elif page == "Batch Analysis":
        page_batch_analysis(model, tokenizer, slang_dict, device)


if __name__ == "__main__":
    main()
