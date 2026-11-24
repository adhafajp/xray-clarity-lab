"""XRay Clarity Lab | dashboard UI/UX for the existing image-processing pipeline.

UI changes are self-contained: algorithms remain in src/noise.py, src/filters.py,
and src/metrics.py. The reference image is only a visual inspiration.
"""
from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, UnidentifiedImageError

from src.filters import run_all_methods
from src.metrics import evaluate_enhancement
from src.noise import apply_noise
from src.sample_generator import create_default_samples

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "data" / "sample_xray"
MAX_UPLOAD_MB = 20
METHODS = ("Median Filter", "Bilateral Filter", "CLAHE")

PALETTES = {
    "light": {
        "bg": "#f5f5fb", "card": "#ffffff", "card2": "#f9f8ff",
        "text": "#23213d", "muted": "#79768f", "line": "#e7e5f2",
        "accent": "#5947d9", "accent2": "#7c67ed", "sidebar": "#302a64",
        "sidebar2": "#211c4e", "chart": "#ffffff", "grid": "#e8e6f1",
        "green": "#18aa8c", "amber": "#efa44b", "blue": "#459bff",
        "switch_track_off": "#14121f", "switch_track_on": "#6954e8",
    },
    "dark": {
        "bg": "#121124", "card": "#211f38", "card2": "#292642",
        "text": "#f7f5ff", "muted": "#b5b0ce", "line": "#393550",
        "accent": "#a69aff", "accent2": "#c0a9ff", "sidebar": "#1a1738",
        "sidebar2": "#121029", "chart": "#211f38", "grid": "#3b3654",
        "green": "#42d3ad", "amber": "#ffbc68", "blue": "#7cb4ff",
        "switch_track_off": "#524c6e", "switch_track_on": "#8e7cff",
    },
}


def load_gray_bytes(raw_bytes: bytes) -> np.ndarray:
    if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Ukuran berkas maksimal {MAX_UPLOAD_MB} MB.")
    try:
        with Image.open(io.BytesIO(raw_bytes)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw_bytes)) as image:
            # Use the same 8-bit grayscale representation as the original project.
            result = np.asarray(image.convert("L"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Berkas tidak dapat dibaca sebagai PNG/JPG yang valid.") from exc
    if result.ndim != 2 or min(result.shape) < 16:
        raise ValueError("Citra harus berukuran minimal 16 × 16 piksel.")
    return result


def load_gray_file(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Citra tidak ditemukan: {path}")
    return image


def get_samples() -> list[Path]:
    files = sorted(p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}) if SAMPLE_DIR.exists() else []
    if not files:
        files = create_default_samples(SAMPLE_DIR)
    return files


def fig_to_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buffer.seek(0)
    return buffer.getvalue()


def png_b64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Gagal mengonversi citra untuk tampilan.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def safe_label(value: object) -> str:
    return escape(str(value), quote=True)


def metric_string(value: float, places: int = 2) -> str:
    if not np.isfinite(value):
        return "∞" if value > 0 else "—"
    return f"{value:.{places}f}"


def apply_dashboard_theme(theme: str) -> None:
    p = PALETTES[theme]
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');
    :root {
      --app-bg: __BG__; --card: __CARD__; --card2: __CARD2__;
      --text: __TEXT__; --muted: __MUTED__; --line: __LINE__;
      --accent: __ACCENT__; --accent2: __ACCENT2__;
      --sidebar: __SIDEBAR__; --sidebar2: __SIDEBAR2__;
      --switch-track-off: __SWITCH_TRACK_OFF__;
      --switch-track-on: __SWITCH_TRACK_ON__;
    }
    html, body, [class*="css"], .stApp {font-family: 'DM Sans',sans-serif;}
    .stApp {background:var(--app-bg); color:var(--text);}
    /* Streamlit's fixed top header overlays widgets when the content padding is too small.
       Reserve enough space so the theme switch remains fully visible at every page refresh. */
    .stApp .block-container {max-width:1500px; padding-top:5rem !important; padding-bottom:3.5rem;}
    [data-testid="stHeader"] {background:var(--app-bg)!important; border-bottom:1px solid var(--line);}
    [data-testid="stHeader"] button {color:var(--text);}
    /* Compact theme switch, kept in the normal document flow instead of floating under the header. */
    .stApp [data-testid="stToggle"] {background:var(--card);border:1px solid var(--line);border-radius:999px;
      padding:8px 12px;box-shadow:0 3px 16px rgba(30,20,90,.035);min-width:145px;}
    .stApp [data-testid="stToggle"] label {margin:0;}
    .stApp [data-testid="stToggle"] label p {color:var(--text)!important;font-size:12px;font-weight:700;white-space:nowrap;}
    .stApp [data-testid="stToggle"] [role="switch"][aria-checked="true"] {background:var(--accent)!important;}
    .stApp h1,.stApp h2,.stApp h3 {font-family:'Outfit','DM Sans',sans-serif; color:var(--text);}
    .stApp p,.stApp label {color:var(--text);}
    [data-testid="stSidebar"] {background:linear-gradient(165deg,var(--sidebar),var(--sidebar2)); border-right:1px solid rgba(255,255,255,.09);}
    [data-testid="stSidebar"] > div:first-child {padding-top:1.3rem;}
    [data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,[data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown {color:#f6f4ff;}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {color:#e9e6ff; font-weight:600;}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#c0bcdd;}
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
      background:rgba(255,255,255,.105)!important; border-color:rgba(255,255,255,.18)!important;
      color:#fff!important; border-radius:12px!important;
    }
    [data-testid="stSidebar"] input {color:#fff!important;}
    [data-testid="stSidebar"] hr {border-color:rgba(255,255,255,.18);}
    [data-testid="stSidebar"] [data-testid="stExpander"] {
      border:1px solid rgba(255,255,255,.17); background:rgba(255,255,255,.05); border-radius:15px;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {color:#fff;}
    .stApp [data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:16px;}
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="input"] > div {
       border-radius:12px; background:var(--card2); color:var(--text);
       border-color:var(--line);
    }
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-baseweb="select"] input,
    .stApp [data-baseweb="input"] input {color:var(--text);}
    .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
       background:var(--card2); border:1px dashed var(--line); border-radius:13px;
    }
    .stApp [data-testid="stDataFrame"] {border:1px solid var(--line);}
    .stApp [data-testid="stToggle"] label p {color:var(--text);}
    .stApp .stTabs [data-baseweb="tab-list"] {
      background:var(--card); padding:7px; border-radius:16px;
      gap:5px; border:1px solid var(--line); box-shadow:0 3px 20px rgba(39,27,100,.03);
    }
    .stApp .stTabs [data-baseweb="tab"] {border-radius:12px; color:var(--muted); font-weight:700; padding:10px 16px;}
    .stApp .stTabs [aria-selected="true"] {background:var(--card2); color:var(--accent)!important;}
    .stApp .stTabs [data-baseweb="tab-highlight"] {background:var(--accent);}
    .stApp .stTabs [data-baseweb="tab-border"] {display:none;}
    .stApp .stButton > button[kind="primary"],
    .stApp .stDownloadButton > button {border-radius:12px; background:linear-gradient(110deg,#6550e1,#8065f6); border:none; color:#fff; font-weight:700;}
    .stApp .stDownloadButton > button:hover {background:linear-gradient(110deg,#4c35ce,#6b50de);color:#fff;}
    .stApp [data-testid="stDataFrame"] {border-radius:16px; overflow:hidden;}
    .stApp [data-testid="stImage"] img {border-radius:14px;}
    [data-testid="stAlert"] {border-radius:14px;}
    .brand-sidebar {display:flex; align-items:center; gap:11px; padding:0 0 13px; color:white;}
    .brand-logo {height:40px;width:40px;display:grid;place-items:center;border-radius:13px;
      background:linear-gradient(140deg,#b6a8ff,#7054f5);box-shadow:0 7px 25px rgba(110,78,240,.35);font:800 17px Outfit,sans-serif;}
    .brand-name {font:700 21px Outfit,sans-serif; letter-spacing:-.6px; line-height:1;}
    .brand-sub {color:#c6c0ee;font-size:10px;letter-spacing:2.3px;font-weight:700;margin-top:5px;}
    .sidebar-group {font-size:11px; letter-spacing:2px; font-weight:800; color:#c9c3f7!important; margin:21px 0 7px;}
    .sidebar-foot {background:rgba(255,255,255,.08);border-radius:14px;padding:13px 15px;margin-top:16px;
      color:#eae6ff;font-size:12px;line-height:1.65;border:1px solid rgba(255,255,255,.1);}
    .topbar {display:flex;align-items:center;justify-content:space-between;margin:2px 0 14px;}
    .topbar .eyebrow {font-size:11px;font-weight:800;letter-spacing:2.3px;color:var(--accent);}
    .topbar .crumb {font-size:12px;color:var(--muted);margin-top:4px;}
    .top-pill {display:inline-block;padding:8px 12px;background:var(--card);border:1px solid var(--line);
      color:var(--muted);border-radius:30px;font-size:11px;font-weight:800;}
    .hero {background:linear-gradient(112deg,#4432a8 0%,#6653d2 62%,#8f72f2 100%);border-radius:24px;
      padding:29px 32px 28px; color:#fff;display:flex;gap:20px;min-height:200px;position:relative;overflow:hidden;
      box-shadow:0 14px 40px rgba(72,48,176,.18);margin-bottom:23px;}
    .hero::before{content:'';position:absolute;width:380px;height:380px;right:-70px;top:-120px;border:1px solid rgba(255,255,255,.21);
      border-radius:50%;box-shadow:0 0 0 35px rgba(255,255,255,.045),0 0 0 80px rgba(255,255,255,.045);}
    .hero-copy {position:relative;z-index:2;flex:1;max-width:780px;}
    .hero-kicker {font-size:11px;letter-spacing:2.2px;font-weight:800;color:#d9d2ff;margin-bottom:12px;}
    .hero h1 {font-family:Outfit,sans-serif;color:#fff!important;font-size:clamp(25px,3vw,37px);line-height:1.15;
      font-weight:750;letter-spacing:-.8px;margin:0 0 11px;}
    .hero-desc {font-size:13px;line-height:1.7;color:#e7e3ff;max-width:620px;margin-bottom:14px;}
    .hero-tags span {display:inline-block;border:1px solid rgba(255,255,255,.23);background:rgba(255,255,255,.12);
      border-radius:30px;padding:6px 11px;color:#fff;font-size:10px;letter-spacing:.4px;font-weight:800;margin:0 7px 0 0;}
    .hero-visual {align-self:center;z-index:2;width:178px;height:148px;border:1px solid rgba(255,255,255,.24);
      border-radius:22px;box-shadow:0 10px 27px rgba(29,20,93,.25);background:rgba(17,10,76,.3);
      overflow:hidden;display:flex;align-items:center;justify-content:center;position:relative;}
    .hero-visual::before{content:'';position:absolute;inset:11px;border:1px dashed rgba(226,222,255,.3);border-radius:14px;}
    .hero-visual img {height:128px;width:128px;object-fit:contain;filter:drop-shadow(0 2px 12px rgba(255,255,255,.14));border-radius:8px;}
    .stats {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px;margin:3px 0 25px;}
    .stat-card {min-height:129px;background:var(--card);border:1px solid var(--line);border-radius:18px;
      padding:18px 18px 14px; box-shadow:0 6px 24px rgba(30,20,90,.035);position:relative;overflow:hidden;}
    .stat-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--stat-accent);}
    .stat-top {display:flex;align-items:center;justify-content:space-between;color:var(--muted);font-size:11px;font-weight:800;letter-spacing:.45px;gap:5px;}
    .stat-symbol {display:grid;place-items:center;border-radius:11px;color:var(--stat-accent);background:var(--card2);font-size:18px;width:32px;height:32px;}
    .stat-value {font-family:Outfit,sans-serif;font-size:25px;font-weight:750;color:var(--text);margin-top:11px;letter-spacing:-.4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    .stat-footer {color:var(--muted);font-size:11px;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    .section-heading {display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin:20px 0 11px;}
    .section-heading h2 {margin:0!important;font:700 22px Outfit,sans-serif!important;color:var(--text)!important;}
    .section-heading p {margin:4px 0 0;color:var(--muted);font-size:12px;}
    .section-chip {border-radius:30px;color:var(--accent);background:var(--card2);border:1px solid var(--line);
      padding:8px 12px;font-size:11px;font-weight:800;white-space:nowrap;}
    .detail-panel {background:var(--card);border:1px solid var(--line);border-radius:17px;padding:17px 17px 8px;margin:6px 0;}
    .detail-panel .detail-eyebrow{font-size:11px;letter-spacing:1.5px;font-weight:800;color:var(--accent);margin-bottom:7px;}
    .detail-panel .detail-title{font-size:19px;font-weight:800;color:var(--text);font-family:Outfit,sans-serif;margin-bottom:7px;}
    .detail-panel .detail-copy{font-size:12px;line-height:1.65;color:var(--muted);}
    .metric-pair {display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:15px;}
    .metric-mini {border-radius:12px;background:var(--card2);padding:11px;min-width:0;}
    .metric-mini span{display:block;color:var(--muted);font-size:10px;font-weight:700;}
    .metric-mini b{font-size:17px;color:var(--text);font-family:Outfit,sans-serif;}
    .note-strip {background:var(--card2);border:1px solid var(--line);padding:12px 15px;border-radius:13px;
      color:var(--muted);font-size:12px;line-height:1.65;margin:13px 0 17px;}
    .gallery-label{font-size:11px;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin:15px 0 8px;}
    .small-foot {font-size:11px;color:var(--muted);margin-top:20px;text-align:center;line-height:1.6;}
    /* ACCESSIBILITY PATCH 2026-11-24: sidebar text is white in BOTH modes.
       Keep these AFTER the generic .stApp widget rules to win the cascade. */
    .stApp [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
    .stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stSidebar"] [data-testid="stRadio"] label p,
    .stApp [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
    .stApp [data-testid="stSidebar"] [data-testid="stSelectbox"] label p,
    .stApp [data-testid="stSidebar"] [data-testid="stSlider"] label p,
    .stApp [data-testid="stSidebar"] [data-testid="stNumberInput"] label p,
    .stApp [data-testid="stSidebar"] [data-testid="stFileUploader"] label p {
      color:#ffffff!important; -webkit-text-fill-color:#ffffff!important; opacity:1!important;
    }
    .stApp [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    .stApp [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span,
    .stApp [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small,
    .stApp [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] div,
    .stApp [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
      color:#eeebff!important; -webkit-text-fill-color:#eeebff!important; opacity:1!important;
    }
    .stApp [data-testid="stSidebar"] [data-baseweb="select"] span,
    .stApp [data-testid="stSidebar"] [data-baseweb="select"] input,
    .stApp [data-testid="stSidebar"] [data-testid="stNumberInput"] input {
      color:#ffffff!important; -webkit-text-fill-color:#ffffff!important;
    }
    /* File uploader: high-contrast violet button and white text on sidebar. */
    .stApp [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
      background:rgba(255,255,255,.10)!important;
      border:1px dashed rgba(225,218,255,.5)!important;
      border-radius:13px!important;
    }
    .stApp [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
    .stApp [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {
      background:#7058e5!important; border:1px solid #b3a5ff!important;
      color:#ffffff!important; -webkit-text-fill-color:#ffffff!important;
      opacity:1!important; font-weight:750!important;
      border-radius:10px!important; box-shadow:0 2px 10px rgba(0,0,0,.12)!important;
    }
    .stApp [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button *,
    .stApp [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] * {
      color:#ffffff!important; -webkit-text-fill-color:#ffffff!important;
      fill:currentColor!important; opacity:1!important;
    }
    .stApp [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
      background:#6042cb!important;border-color:#d6ceff!important;
    }
    /* Visible theme switch: black OFF track in light, lavender ON in dark.
       Multiple selectors cover newer/older Streamlit/BaseWeb markup. */
    .stApp [data-testid="stToggle"] input[role="switch"] + div,
    .stApp [data-testid="stToggle"] [role="switch"][aria-checked="false"],
    .stApp [data-testid="stToggle"] label[data-baseweb="checkbox"] > span:first-of-type,
    .stApp [data-testid="stToggle"] label[data-baseweb="checkbox"] > div:first-of-type {
      background-color:var(--switch-track-off)!important;
      border-color:var(--switch-track-off)!important;
      border-radius:999px!important;
      box-shadow:0 0 0 1px rgba(255,255,255,.15)!important;
      opacity:1!important;
    }
    .stApp [data-testid="stToggle"] input[role="switch"]:checked + div,
    .stApp [data-testid="stToggle"] [role="switch"][aria-checked="true"],
    .stApp [data-testid="stToggle"] label[data-baseweb="checkbox"]:has(input:checked) > span:first-of-type,
    .stApp [data-testid="stToggle"] label[data-baseweb="checkbox"]:has(input:checked) > div:first-of-type {
      background-color:var(--switch-track-on)!important;
      border-color:var(--switch-track-on)!important;
      box-shadow:0 0 0 1px rgba(174,160,255,.44)!important;
      opacity:1!important;
    }
    .stApp [data-testid="stToggle"] input[role="switch"] + div > div,
    .stApp [data-testid="stToggle"] [role="switch"]::before {
      background:#ffffff!important;
    }
    .stApp [data-testid="stToggle"] label p,
    .stApp [data-testid="stToggle"] label span:last-child {
      color:var(--text)!important; -webkit-text-fill-color:var(--text)!important;
      opacity:1!important;
    }
    @media(max-width:1000px){.stats{grid-template-columns:repeat(2,minmax(0,1fr));}.hero-visual{width:135px;height:128px;}.hero-visual img{width:105px;height:105px}}
    @media(max-width:650px){.stApp .block-container{padding-top:4.75rem!important;}
      .topbar .crumb{font-size:10px;line-height:1.4;}
      .stApp [data-testid="stToggle"]{padding:7px 9px;min-width:0;}
      .stApp [data-testid="stToggle"] label p{font-size:11px;}
      .stats{grid-template-columns:1fr 1fr;gap:9px;}.stat-card{padding:12px;min-height:112px;}.stat-value{font-size:19px;}
      .hero{padding:23px 19px;}.hero-visual{display:none;}.block-container{padding-left:1rem;padding-right:1rem;}.top-pill{display:none;}}
    </style>
    """
    for k, v in p.items():
        css = css.replace("__" + k.upper() + "__", v)
    st.markdown(css, unsafe_allow_html=True)


def heading(title: str, subtitle: str, chip: str = "") -> None:
    optional = f'<span class="section-chip">{safe_label(chip)}</span>' if chip else ""
    st.markdown(
        f'<div class="section-heading"><div><h2>{safe_label(title)}</h2>'
        f'<p>{safe_label(subtitle)}</p></div>{optional}</div>',
        unsafe_allow_html=True,
    )


def dashboard_toolbar() -> None:
    """Display branding and the real Streamlit theme control in one visible row."""
    title_col, theme_col = st.columns([5, 1.3], gap="medium", vertical_alignment="center")
    with title_col:
        st.markdown(
            '<div class="topbar"><div><div class="eyebrow">XRAY CLARITY LAB / DASHBOARD</div>'
            '<div class="crumb">Workspace &nbsp;/&nbsp; Pengolahan Citra &nbsp;/&nbsp; Komparasi Filter</div>'
            '</div><span class="top-pill">● &nbsp; EXPERIMENTAL WORKSPACE</span></div>',
            unsafe_allow_html=True,
        )
    with theme_col:
        # Label describes the mode that can be activated, not the current mode.
        # Streamlit reruns this function when the toggle value changes.
        next_mode_label = "☀️ Mode Terang" if st.session_state["dark_mode"] else "🌙 Mode Gelap"
        st.toggle(
            next_mode_label, key="dark_mode",
            help="Klik untuk beralih antara tampilan terang dan gelap.",
        )


def dashboard_hero(ref_image: np.ndarray) -> None:
    visual = png_b64(ref_image)
    st.markdown(
        '<div class="hero"><div class="hero-copy">'
        '<div class="hero-kicker">IMAGE ENHANCEMENT STUDIO &nbsp; / &nbsp; 2026</div>'
        '<h1>Citra lebih jernih.<br>Analisis lebih terukur.</h1>'
        '<div class="hero-desc">Bandingkan Median, Bilateral, dan CLAHE pada citra X-Ray berderau. '
        'Atur parameter, geser Before–After, dan evaluasi hasil secara instan.</div>'
        '<div class="hero-tags"><span>03 METODE</span><span>02 JENIS DERAU</span>'
        '<span>RISET NON-KLINIS</span></div></div>'
        f'<div class="hero-visual"><img src="data:image/png;base64,{visual}" alt="Preview citra uji"></div></div>',
        unsafe_allow_html=True,
    )


def stat_card(title: str, value: str, footer: str, glyph: str, accent: str) -> str:
    return (
        f'<div class="stat-card" style="--stat-accent:{accent}"><div class="stat-top">'
        f'<span>{safe_label(title)}</span><span class="stat-symbol">{safe_label(glyph)}</span></div>'
        f'<div class="stat-value" title="{safe_label(value)}">{safe_label(value)}</div>'
        f'<div class="stat-footer" title="{safe_label(footer)}">{safe_label(footer)}</div></div>'
    )


def render_stats(ref: np.ndarray, img_name: str, df: pd.DataFrame, theme: str) -> None:
    p = PALETTES[theme]
    only_methods = df[df["Metode"].isin(METHODS)]
    best_psnr = only_methods.loc[only_methods["PSNR (dB)"].idxmax()]
    best_ssim = only_methods.loc[only_methods["SSIM"].idxmax()]
    fastest = only_methods.loc[only_methods["Waktu (ms)"].idxmin()]
    cards = [
        stat_card("CITRA UJI", f"{ref.shape[1]} × {ref.shape[0]}", img_name, "▣", p["accent"]),
        stat_card("PSNR TERTINGGI", f"{metric_string(best_psnr['PSNR (dB)'])} dB", best_psnr["Metode"], "↗", p["blue"]),
        stat_card("SSIM TERTINGGI", metric_string(best_ssim["SSIM"], 4), best_ssim["Metode"], "◉", p["green"]),
        stat_card("PROSES TERCEPAT", f"{metric_string(fastest['Waktu (ms)'])} ms", fastest["Metode"], "⚡", p["amber"]),
    ]
    st.markdown('<div class="stats">' + "".join(cards) + '</div>', unsafe_allow_html=True)


def make_comparison_figure(ref, noisy, outputs, metrics_map, theme="light"):
    p = PALETTES[theme]
    items = [
        ("Citra Asli", ref, None),
        ("Berderau", noisy, metrics_map["Citra Berderau"]),
        *[(method, outputs[method]["image"], metrics_map[method]) for method in METHODS],
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), dpi=130)
    fig.patch.set_facecolor(p["chart"])
    for ax, (label, img, m) in zip(axes, items):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        ax.set_facecolor(p["chart"])
        ax.set_title(label if m is None else
                     f"{label}\nPSNR {metric_string(m['PSNR (dB)'])} dB | SSIM {m['SSIM']:.4f}",
                     fontsize=10 if m is None else 8.5, fontweight="bold", color=p["text"], pad=10)
    fig.tight_layout(pad=1.4)
    return fig


def make_metrics_chart(df, theme="light"):
    p = PALETTES[theme]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1), dpi=125)
    fig.patch.set_facecolor(p["chart"])
    colors = [p["muted"], "#6e5af2", "#38b5aa", "#f0a552"]
    labels = [("PSNR (dB)", "PSNR · Higher is better"),
              ("SSIM", "SSIM · Higher is better"),
              ("Waktu (ms)", "Processing time · Lower is better")]
    for ax, (column, title) in zip(axes, labels):
        ax.set_facecolor(p["chart"])
        raw = df[column].astype(float).to_numpy()
        finite = raw[np.isfinite(raw)]
        ceiling = max(1.0, float(np.max(finite)) * 1.20) if len(finite) else 1.0
        heights = [ceiling * .90 if (np.isposinf(v)) else max(0.0, v) for v in raw]
        bars = ax.bar(range(len(df)), heights, color=colors[:len(df)], width=.58, zorder=3)
        ax.set_title(title, fontsize=11, color=p["text"], fontweight="bold", pad=18)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([name.replace(" Filter", "").replace("Citra Berderau", "Berderau")
                            for name in df["Metode"]], rotation=16, ha="right", fontsize=8.5)
        ax.set_ylim(0, ceiling * 1.13)
        ax.tick_params(axis="both", colors=p["muted"], labelsize=8)
        ax.grid(axis="y", color=p["grid"], alpha=.85, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for bar, val in zip(bars, raw):
            label = metric_string(val, 4 if column == "SSIM" else 2)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ceiling * .02,
                    label, ha="center", va="bottom", fontsize=8, color=p["text"])
    fig.tight_layout(pad=2.2)
    return fig


def make_histogram_chart(ref, noisy, outputs, theme="light"):
    p = PALETTES[theme]
    fig, ax = plt.subplots(figsize=(12.5, 4.3), dpi=125)
    fig.patch.set_facecolor(p["chart"])
    ax.set_facecolor(p["chart"])
    series = [
        ("Citra Asli", ref, p["text"], "-"),
        ("Berderau", noisy, p["muted"], "--"),
        ("Median Filter", outputs["Median Filter"]["image"], "#7465f5", "-"),
        ("Bilateral Filter", outputs["Bilateral Filter"]["image"], "#36b8aa", "-"),
        ("CLAHE", outputs["CLAHE"]["image"], "#f0a453", "-"),
    ]
    for name, img, color, style in series:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
        ax.plot(hist, label=name, color=color, linestyle=style, linewidth=1.8)
    ax.set_title("Distribusi intensitas piksel", color=p["text"], fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Intensitas · 0–255", color=p["muted"])
    ax.set_ylabel("Jumlah piksel", color=p["muted"])
    ax.set_xlim(0, 255)
    ax.grid(alpha=.8, color=p["grid"], linestyle="--")
    ax.tick_params(colors=p["muted"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=9)
    for text in legend.get_texts():
        text.set_color(p["text"])
    fig.tight_layout(pad=1.8)
    return fig


def before_after_view(before: np.ndarray, after: np.ndarray,
                      before_label: str, after_label: str, theme: str) -> None:
    """A standalone interactive wipe slider, using Streamlit's supported HTML component."""
    if before.shape != after.shape:
        raise ValueError("Perbandingan hanya mendukung gambar dengan dimensi sama.")
    p = PALETTES[theme]
    left = png_b64(before)
    right = png_b64(after)
    before_label, after_label = safe_label(before_label), safe_label(after_label)
    html = '''<!DOCTYPE html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  *{box-sizing:border-box} body{margin:0;font-family:Arial,sans-serif;background:transparent}
  .frame{position:relative;width:100%;height:383px;background:#0d0d1b;border-radius:17px;overflow:hidden;
     border:1px solid __BORDER__;user-select:none;--split:50%}
  .frame img{position:absolute;width:100%;height:100%;object-fit:contain;inset:0;pointer-events:none}
  .frame .after{clip-path:inset(0 calc(100% - var(--split)) 0 0)}
  .frame .line{position:absolute;top:0;bottom:0;left:var(--split);width:2px;background:#fff;
     transform:translateX(-50%);pointer-events:none;box-shadow:0 0 11px rgba(0,0,0,.3)}
  .handle{position:absolute;left:var(--split);top:50%;transform:translate(-50%,-50%);pointer-events:none;
     width:42px;height:42px;display:grid;place-items:center;background:#fff;border-radius:50%;
     color:#4c3ac2;font-size:22px;font-weight:bold;box-shadow:0 4px 22px rgba(0,0,0,.3)}
  .badge{position:absolute;top:16px;z-index:4;color:#fff;background:rgba(18,12,46,.72);border:1px solid rgba(255,255,255,.22);
     backdrop-filter:blur(8px);padding:9px 13px;border-radius:10px;font-size:11px;font-weight:700;max-width:45%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .badge.left{left:15px}.badge.right{right:15px}
  .slider{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;z-index:5;margin:0}
  .below{display:flex;justify-content:space-between;align-items:center;margin:11px 3px;color:__MUTED__;font-size:12px;font-weight:700;gap:10px}
  .below .tip{font-weight:400;text-align:center;flex:1}
  @media(max-width:600px){.frame{height:320px}.badge{font-size:10px;padding:7px}}
</style></head><body>
<div class="frame" id="comparison">
  <img src="data:image/png;base64,__BEFORE__" alt="Sebelum" />
  <img class="after" src="data:image/png;base64,__AFTER__" alt="Sesudah" />
  <div class="line"></div><div class="handle">‹ ›</div>
  <div class="badge left">BEFORE · __BEFORE_LABEL__</div><div class="badge right">AFTER · __AFTER_LABEL__</div>
  <input id="wipe" class="slider" type="range" min="0" max="100" value="50" step="1"
         aria-label="Geser untuk membandingkan sebelum dan sesudah" />
</div>
<div class="below"><span>◀ SEBELUM</span><span class="tip">Geser pembatas atau gunakan tombol panah keyboard</span><span>SESUDAH ▶</span></div>
<script>
const range = document.querySelector('#wipe');
const frame = document.querySelector('#comparison');
range.addEventListener('input', () => frame.style.setProperty('--split',range.value+'%'));
</script></body></html>'''
    for marker, val in {
        "__BORDER__": p["line"], "__MUTED__": p["muted"],
        "__BEFORE__": left, "__AFTER__": right,
        "__BEFORE_LABEL__": before_label, "__AFTER_LABEL__": after_label,
    }.items():
        html = html.replace(marker, val)
    components.html(html, height=433, scrolling=False)


def selected_metrics_panel(method: str, metrics: dict, noise: str) -> None:
    descriptions = {
        "Median Filter": "Mengurangi derau impuls dengan nilai median piksel di sekitar titik yang diproses.",
        "Bilateral Filter": "Menghaluskan citra sambil mempertimbangkan jarak spasial dan perbedaan intensitas piksel.",
        "CLAHE": "Meningkatkan kontras lokal. Bukan metode denoising murni; PSNR rendah tidak selalu berarti implementasi salah.",
    }
    data = metrics[method]
    boxes = [
        ("PSNR", metric_string(data["PSNR (dB)"]) + " dB"),
        ("SSIM", metric_string(data["SSIM"], 4)),
        ("MSE", metric_string(data["MSE"])),
        ("WAKTU", metric_string(data["Waktu (ms)"]) + " ms"),
    ]
    mini = "".join(f'<div class="metric-mini"><span>{safe_label(name)}</span>'
                   f'<b>{safe_label(val)}</b></div>' for name, val in boxes)
    st.markdown(
        f'<div class="detail-panel"><div class="detail-eyebrow">FILTER YANG DIPILIH</div>'
        f'<div class="detail-title">{safe_label(method)}</div>'
        f'<div class="detail-copy">{safe_label(descriptions[method])}</div>'
        f'<div class="metric-pair">{mini}</div>'
        f'<div class="note-strip">Perhitungan menggunakan citra input sebelum derau buatan sebagai referensi. '
        f'Tampilan ini untuk evaluasi teknis, bukan diagnosis medis.</div></div>',
        unsafe_allow_html=True,
    )


def sidebar_controls(samples: list[Path]) -> dict | None:
    with st.sidebar:
        st.markdown(
            '<div class="brand-sidebar"><div class="brand-logo">XR</div>'
            '<div><div class="brand-name">clarity<span style="color:#ac9cff">lab</span></div>'
            '<div class="brand-sub">X-RAY IMAGE STUDIO</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-group">01 &nbsp; / &nbsp; SUMBER CITRA</div>', unsafe_allow_html=True)
        input_mode = st.radio("Sumber citra", ["Citra Contoh", "Upload Sendiri"], label_visibility="collapsed")
        uploaded, chosen_sample = None, None
        if input_mode == "Upload Sendiri":
            uploaded = st.file_uploader("Upload X-Ray (PNG/JPG)", type=["png", "jpg", "jpeg"],
                                        help="Maksimum 20 MB per file. Citra dikonversi ke grayscale 8-bit.")
        else:
            chosen_sample = st.selectbox("Pilih citra contoh", samples,
                                         format_func=lambda path: path.stem.replace("sample_xray_", "").replace("_", " ").title())
        st.markdown('<div class="sidebar-group">02 &nbsp; / &nbsp; SIMULASI DERAU</div>', unsafe_allow_html=True)
        noise_name = st.selectbox("Jenis derau", ["Gaussian", "Salt-and-Pepper", "Tanpa Derau"])
        noise_map = {"Gaussian": "gaussian", "Salt-and-Pepper": "salt_pepper", "Tanpa Derau": "none"}
        noise_type = noise_map[noise_name]
        if noise_type == "gaussian":
            noise_level = st.slider("Intensitas · Sigma", 5.0, 60.0, 20.0, 1.0)
        elif noise_type == "salt_pepper":
            noise_level = st.slider("Rasio piksel berderau", 0.01, 0.25, 0.05, 0.01, format="%.2f")
        else:
            noise_level = 0.0
            st.caption("Tidak ada derau sintetis yang ditambahkan.")
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1,
                               help="Gunakan seed tetap agar hasil derau dapat direproduksi.")
        st.markdown('<div class="sidebar-group">03 &nbsp; / &nbsp; PARAMETER FILTER</div>', unsafe_allow_html=True)
        with st.expander("Median Filter", expanded=False):
            median_k = st.select_slider("Ukuran kernel (ganjil)", options=[3, 5, 7, 9, 11], value=5)
        with st.expander("Bilateral Filter", expanded=False):
            bilat_d = st.select_slider("Diameter", options=[5, 7, 9, 11, 15], value=9)
            bilat_sc = st.slider("Sigma color", 10.0, 150.0, 75.0, 5.0)
            bilat_ss = st.slider("Sigma space", 10.0, 150.0, 75.0, 5.0)
        with st.expander("CLAHE", expanded=False):
            clahe_clip = st.slider("Clip limit", 1.0, 8.0, 2.0, 0.5)
            clahe_grid = st.select_slider("Tile grid", options=[4, 8, 12, 16], value=8)
        st.markdown(
            '<div class="sidebar-foot">✦ <b>Catatan penelitian</b><br>'
            'Aplikasi membandingkan kualitas citra secara komputasional. '
            'Jangan gunakan hasilnya sebagai dasar diagnosis klinis.</div>',
            unsafe_allow_html=True,
        )
    if input_mode == "Upload Sendiri" and uploaded is None:
        return None
    try:
        if uploaded is not None:
            ref_img = load_gray_bytes(uploaded.getvalue())
            img_name = uploaded.name
        else:
            ref_img = load_gray_file(chosen_sample)
            img_name = chosen_sample.name
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
        return None
    return {
        "ref": ref_img, "img_name": img_name, "noise_type": noise_type,
        "noise_name": noise_name, "noise_level": noise_level, "seed": int(seed),
        "filter_params": dict(
            median_ksize=median_k, bilateral_d=bilat_d,
            bilateral_sigma_color=bilat_sc, bilateral_sigma_space=bilat_ss,
            clahe_clip_limit=clahe_clip, clahe_grid_size=clahe_grid,
        ),
    }


def render_visual_tab(ref, noisy, outputs, metrics_map, img_name, noise_type, theme):
    heading("Before & After Studio", "Geser untuk melihat perubahan filter pada gambar yang sama.", "LIVE PREVIEW")
    before_options = {"Citra Berderau": noisy, "Citra Asli": ref}
    controls_left, controls_right = st.columns(2)
    with controls_left:
        before_name = st.selectbox("Bandingkan dari", list(before_options), index=0)
    with controls_right:
        selected = st.selectbox("Hasil metode", list(METHODS), index=1)
    left, right = st.columns([1.73, 1.0], gap="medium")
    with left:
        before_after_view(before_options[before_name], outputs[selected]["image"], before_name, selected, theme)
    with right:
        selected_metrics_panel(selected, metrics_map, noise_type)
    heading("Galeri Hasil Pemrosesan", "Lima visual disusun agar perubahan tekstur dan kontras mudah dibandingkan.")
    gallery = [
        ("Citra Asli", ref), ("Citra Berderau", noisy),
        *[(name, outputs[name]["image"]) for name in METHODS],
    ]
    cols = st.columns(5, gap="small")
    for col, (title, img) in zip(cols, gallery):
        with col:
            st.markdown(f'<div class="gallery-label">{safe_label(title)}</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True, clamp=True)
    fig = make_comparison_figure(ref, noisy, outputs, metrics_map, theme)
    try:
        st.download_button("↓ Unduh kolase perbandingan (PNG)", fig_to_bytes(fig),
                           file_name=f"hasil_{Path(img_name).stem}_{noise_type}.png",
                           mime="image/png", key="download_comparison")
    finally:
        plt.close(fig)


def render_metrics_tab(df, img_name, noise_type, theme):
    heading("Performance Analytics", "PSNR, SSIM, MSE, dan waktu pemrosesan dari seluruh metode.", "AUTO EVALUATION")
    st.markdown('<div class="note-strip">PSNR dan SSIM lebih tinggi biasanya menunjukkan kesamaan lebih besar dengan '
                'citra referensi. MSE dan waktu proses lebih rendah umumnya lebih baik. '
                'CLAHE berfokus pada kontras lokal, sehingga skornya perlu ditafsirkan sesuai tujuan.</div>',
                unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    fig = make_metrics_chart(df, theme)
    try:
        st.pyplot(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("↓ Ekspor data metrik (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                               file_name=f"metrik_{Path(img_name).stem}_{noise_type}.csv", mime="text/csv")
        with c2:
            st.download_button("↓ Ekspor grafik (PNG)", fig_to_bytes(fig),
                               file_name=f"grafik_{Path(img_name).stem}_{noise_type}.png", mime="image/png")
    finally:
        plt.close(fig)


def render_histogram_tab(ref, noisy, outputs, img_name, noise_type, theme):
    heading("Histogram Explorer", "Amati persebaran intensitas 0–255 dari citra asli dan setiap hasil filter.", "PIXEL DISTRIBUTION")
    fig = make_histogram_chart(ref, noisy, outputs, theme)
    try:
        st.pyplot(fig, use_container_width=True)
        st.download_button("↓ Unduh histogram (PNG)", fig_to_bytes(fig),
                           file_name=f"histogram_{Path(img_name).stem}_{noise_type}.png", mime="image/png")
    finally:
        plt.close(fig)
    st.markdown('<div class="note-strip">Histogram menunjukkan distribusi intensitas, bukan informasi diagnostik. '
                'Perubahan sebaran setelah CLAHE terutama mencerminkan peningkatan kontras lokal.</div>',
                unsafe_allow_html=True)


def render_batch_tab(cfg, theme):
    heading("Batch Evaluation", "Uji beberapa citra dengan parameter dan random seed yang sama.", "MULTI-IMAGE")
    st.markdown('<div class="note-strip">Setiap citra asli diperlakukan sebagai referensi sebelum penambahan '
                'derau sintetis. Rata-rata dan standar deviasi ditampilkan per metode. '
                'Nama dan asal dataset harus didokumentasikan secara terpisah.</div>', unsafe_allow_html=True)
    batch_files = st.file_uploader("Pilih beberapa citra PNG/JPG", type=["png", "jpg", "jpeg"],
                                   accept_multiple_files=True, key="batch_files")
    if not batch_files:
        st.info("Tambahkan dua atau lebih citra untuk mulai membandingkan hasil batch.")
        return
    if st.button("▶ Jalankan evaluasi batch", type="primary", key="run_batch"):
        rows, issues = [], []
        progress = st.progress(0)
        for i, f in enumerate(batch_files):
            try:
                original = load_gray_bytes(f.getvalue())
                noisy = apply_noise(original, noise_type=cfg["noise_type"],
                                    intensity=cfg["noise_level"], seed=cfg["seed"])
                outputs = run_all_methods(noisy, **cfg["filter_params"])
                rows.append({"Citra": f.name, "Metode": "Citra Berderau",
                             **evaluate_enhancement(original, noisy, 0.0)})
                for method, result in outputs.items():
                    rows.append({"Citra": f.name, "Metode": method,
                                 **evaluate_enhancement(original, result["image"], result["elapsed_ms"])})
            except (ValueError, cv2.error) as exc:
                issues.append(f"{f.name}: {exc}")
            progress.progress((i + 1) / len(batch_files))
        st.session_state["batch_result"] = {"rows": rows, "errors": issues,
                                             "signature": (cfg["noise_type"], cfg["noise_level"],
                                                           cfg["seed"], tuple(cfg["filter_params"].items()),
                                                           tuple((f.name, f.size) for f in batch_files))}
    result = st.session_state.get("batch_result")
    signature = (cfg["noise_type"], cfg["noise_level"], cfg["seed"],
                 tuple(cfg["filter_params"].items()), tuple((f.name, f.size) for f in batch_files))
    if result and result["signature"] != signature:
        st.info("Parameter atau daftar berkas berubah. Jalankan evaluasi batch kembali untuk hasil terbaru.")
        return
    if not result:
        return
    for error in result["errors"]:
        st.warning(error)
    if not result["rows"]:
        st.error("Tidak ada citra yang berhasil diproses.")
        return
    detail = pd.DataFrame(result["rows"])
    summary = detail.groupby("Metode", sort=False)[["PSNR (dB)", "SSIM", "MSE", "Waktu (ms)"]].agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    summary = summary.fillna({col: 0 for col in summary.columns if col.endswith("_std")}).round(4).reset_index()
    st.success(f"Berhasil mengevaluasi {detail['Citra'].nunique()} citra.")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    with st.expander("Lihat hasil untuk setiap citra"):
        st.dataframe(detail, use_container_width=True, hide_index=True)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("↓ Unduh ringkasan batch (CSV)", summary.to_csv(index=False).encode("utf-8"),
                           file_name=f"batch_summary_{cfg['noise_type']}.csv", mime="text/csv")
    with d2:
        st.download_button("↓ Unduh detail per citra (CSV)", detail.to_csv(index=False).encode("utf-8"),
                           file_name=f"batch_detail_{cfg['noise_type']}.csv", mime="text/csv")


def main():
    st.set_page_config(page_title="XRay Clarity Lab | Image Studio", page_icon="🔬",
                       layout="wide", initial_sidebar_state="expanded")
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    theme = "dark" if st.session_state["dark_mode"] else "light"
    apply_dashboard_theme(theme)
    samples = get_samples()
    cfg = sidebar_controls(samples)

    dashboard_toolbar()
    if cfg is None:
        heading("Mulai dari citra X-Ray", "Unggah citra PNG/JPG melalui sidebar untuk membuka dashboard analisis.")
        st.info("Belum ada citra yang dipilih. Kamu juga dapat menggunakan Citra Contoh dari sidebar.")
        return

    ref = cfg["ref"]
    noisy = apply_noise(ref, noise_type=cfg["noise_type"],
                        intensity=cfg["noise_level"], seed=cfg["seed"])
    outputs = run_all_methods(noisy, **cfg["filter_params"])
    metrics = {"Citra Berderau": evaluate_enhancement(ref, noisy, 0.0)}
    for method, result in outputs.items():
        metrics[method] = evaluate_enhancement(ref, result["image"], result["elapsed_ms"])
    df = pd.DataFrame([{"Metode": method, **value} for method, value in metrics.items()])
    dashboard_hero(ref)
    render_stats(ref, cfg["img_name"], df, theme)
    tab_v, tab_m, tab_h, tab_b = st.tabs([
        "◈  Visual Studio", "▥  Metrik & Grafik", "⌁  Histogram", "▦  Batch Uji",
    ])
    with tab_v:
        render_visual_tab(ref, noisy, outputs, metrics, cfg["img_name"], cfg["noise_type"], theme)
    with tab_m:
        render_metrics_tab(df, cfg["img_name"], cfg["noise_type"], theme)
    with tab_h:
        render_histogram_tab(ref, noisy, outputs, cfg["img_name"], cfg["noise_type"], theme)
    with tab_b:
        render_batch_tab(cfg, theme)
    st.markdown('<div class="small-foot">XRAY CLARITY LAB &nbsp;·&nbsp; Tugas Pengolahan Citra Digital'
                ' &nbsp;·&nbsp; Untuk penelitian non-klinis / pembelajaran</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
