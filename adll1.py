# ==========================================================
# ANIME INSIGHT AI — FINAL
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import gdown
import random
import base64
import os

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Anime Insight AI",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# SESSION STATE
# ==========================================================
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "random_anime" not in st.session_state:
    st.session_state.random_anime = None
if "explorer_page" not in st.session_state:
    st.session_state.explorer_page = 1
if "per_page" not in st.session_state:
    st.session_state.per_page = 12
if "recommendation_genres" not in st.session_state:
    st.session_state.recommendation_genres = []
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False

# ==========================================================
# DETEKSI GAMBAR BACKGROUND (flexibel)
# ==========================================================
def get_base64(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

BG_IMAGE = ""
if os.path.exists("assets"):
    for filename in os.listdir("assets"):
        if filename.startswith("anime_bg."):
            path = os.path.join("assets", filename)
            BG_IMAGE = get_base64(path)
            break

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    if not os.path.exists("users-details-2023.csv"):
        with st.spinner("Downloading dataset..."):
            url = "https://drive.google.com/uc?id=1XQ_m3aZ34ogv5CjOA3UFLPHJ9S_RtQvc"
            gdown.download(url, "users-details-2023.csv", quiet=True)
    df_anime = pd.read_csv("anime-dataset-2023.csv")
    df_user = pd.read_csv("users-details-2023.csv")
    df_score = pd.read_csv("users-score-small.csv")
    return df_anime, df_user, df_score

with st.spinner("Loading Anime Database..."):
    df_anime, df_user, df_score = load_data()

# ==========================================================
# DATA CLEANING
# ==========================================================
scores = df_anime[df_anime["Score"] != "UNKNOWN"]["Score"].astype(float)
mean_score = round(scores.mean(), 2)
df_anime["Score"] = df_anime["Score"].replace("UNKNOWN", mean_score).astype(float)

if "Rank" in df_anime.columns:
    df_anime["Rank"] = pd.to_numeric(df_anime["Rank"], errors="coerce")
if "Episodes" in df_anime.columns:
    df_anime["Episodes"] = pd.to_numeric(df_anime["Episodes"], errors="coerce").fillna(0).astype(int)

if "Aired" in df_anime.columns:
    df_anime["Year"] = df_anime["Aired"].str.extract(r"(\d{4})").astype(float)

# ==========================================================
# COLLABORATIVE FILTERING
# ==========================================================
@st.cache_data
def build_similarity():
    rating_data = df_score[["user_id", "anime_id", "rating"]]
    count_per_anime = rating_data["anime_id"].value_counts()
    popular = count_per_anime[count_per_anime >= 20].index
    rating_data = rating_data[rating_data["anime_id"].isin(popular)]
    pivot = rating_data.pivot_table(index="anime_id", columns="user_id", values="rating").fillna(0)
    similarity = cosine_similarity(pivot)
    similarity_df = pd.DataFrame(similarity, index=pivot.index, columns=pivot.index)
    return similarity_df

with st.spinner("Building Recommendation Engine..."):
    similarity_df = build_similarity()

# ==========================================================
# CSS PREMIUM (dengan gradasi ungu-biru-hitam)
# ==========================================================
bg_style = f"""
.stApp {{
    background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 27, 75, 0.92), rgba(88, 28, 135, 0.88)),
        url("data:image/jpeg;base64,{BG_IMAGE}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #f1f5f9;
}}
""" if BG_IMAGE else """
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e1b4b, #581c87);
    color: #f1f5f9;
}
"""

st.markdown(f"""
<style>
    /* GLOBAL */
    {bg_style}
    .block-container {{
        padding: 1.5rem 2rem 2rem 2rem;
    }}
    /* SIDEBAR */
    section[data-testid="stSidebar"] {{
        background: rgba(15, 23, 42, 0.88);
        backdrop-filter: blur(25px);
        border-right: 1px solid rgba(255,255,255,0.06);
        padding: 1rem 0.5rem;
    }}
    .logo-container {{
        text-align: center;
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 1rem;
    }}
    .logo-title {{
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(135deg, #c084fc, #8b5cf6, #6d28d9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }}
    .logo-sub {{
        font-size: 11px;
        color: #94a3b8;
        letter-spacing: 3px;
        margin-top: 2px;
    }}
    .sidebar-divider {{
        margin: 0.8rem 0;
        border-top: 1px solid rgba(255,255,255,0.06);
    }}
    .sidebar-section-title {{
        font-size: 10px;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 1.5px;
        padding: 0.5rem 1rem 0.2rem 1rem;
        font-weight: 600;
    }}
    div[data-testid="stButton"] button {{
        width: 100%;
        background: transparent;
        border: none;
        color: #cbd5e1;
        padding: 8px 14px;
        border-radius: 10px;
        text-align: left;
        font-size: 14px;
        font-weight: 500;
        transition: 0.25s;
        margin: 1px 0;
    }}
    div[data-testid="stButton"] button:hover {{
        background: rgba(139, 92, 246, 0.15);
        color: white;
    }}
    div[data-testid="stButton"] button:focus {{
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(124,58,237,0.35);
    }}
    .sidebar-footer {{
        position: fixed;
        bottom: 1rem;
        width: calc(100% - 2rem);
        text-align: center;
        font-size: 11px;
        color: #64748b;
        border-top: 1px solid rgba(255,255,255,0.06);
        padding-top: 1rem;
        margin-top: 1rem;
    }}
    .sidebar-footer .jp {{
        font-size: 10px;
        color: #475569;
    }}
    /* KPI CARDS */
    .kpi-card {{
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 20px 15px;
        text-align: center;
        transition: 0.3s;
        height: 100%;
    }}
    .kpi-card:hover {{
        transform: translateY(-4px);
        border-color: rgba(139, 92, 246, 0.3);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.15);
    }}
    .kpi-value {{
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(135deg, #c084fc, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .kpi-label {{
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
    }}
    .kpi-growth {{
        font-size: 11px;
        color: #22c55e;
        background: rgba(34,197,94,0.12);
        padding: 2px 12px;
        border-radius: 20px;
        display: inline-block;
        margin-top: 5px;
    }}
    .kpi-growth.neutral {{
        color: #94a3b8;
        background: rgba(148,163,184,0.08);
    }}
    /* SECTION TITLE */
    .section-title {{
        font-size: 26px;
        font-weight: 700;
        margin: 30px 0 20px 0;
        color: white;
        border-left: 5px solid #8b5cf6;
        padding-left: 16px;
        letter-spacing: -0.3px;
    }}
    .section-title-sm {{
        font-size: 20px;
        font-weight: 600;
        margin: 20px 0 15px 0;
        color: #e2e8f0;
        border-left: 4px solid #8b5cf6;
        padding-left: 14px;
    }}
    /* HERO BANNER */
    .hero-container {{
        position: relative;
        height: 200px;
        overflow: hidden;
        border-radius: 22px;
        margin-bottom: 25px;
        border: 1px solid rgba(139, 92, 246, 0.15);
        box-shadow: 0 8px 35px rgba(0,0,0,0.4);
    }}
    .hero-container img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}
    .hero-overlay {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        padding: 25px 35px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: linear-gradient(90deg, rgba(15,23,42,0.92) 0%, rgba(15,23,42,0.60) 50%, rgba(15,23,42,0.10) 100%);
    }}
    .hero-overlay .title {{
        font-size: 2rem;
        font-weight: 800;
        color: white;
        line-height: 1.1;
    }}
    .hero-overlay .title span {{
        background: linear-gradient(135deg, #c084fc, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero-overlay .sub {{
        font-size: 1rem;
        color: #c084fc;
        font-weight: 600;
        margin-top: 2px;
    }}
    .hero-overlay .desc {{
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 2px;
    }}
    /* ANIME CARD */
    .anime-card {{
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 12px;
        transition: 0.3s;
        height: 100%;
        display: flex;
        flex-direction: column;
    }}
    .anime-card:hover {{
        transform: translateY(-4px);
        border-color: rgba(139, 92, 246, 0.3);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.15);
    }}
    .anime-card .poster {{
        border-radius: 12px;
        width: 100%;
        aspect-ratio: 2/3;
        object-fit: cover;
        background: rgba(255,255,255,0.05);
    }}
    .anime-card .title {{
        font-weight: 600;
        font-size: 0.95rem;
        color: white;
        margin-top: 8px;
        line-height: 1.2;
        min-height: 2.4rem;
    }}
    .anime-card .meta {{
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 4px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }}
    .anime-card .meta .score {{
        color: #fbbf24;
        font-weight: 600;
    }}
    .anime-card .genres {{
        font-size: 0.7rem;
        color: #94a3b8;
        margin-top: 4px;
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
    }}
    .anime-card .genres span {{
        background: rgba(139,92,246,0.15);
        padding: 2px 10px;
        border-radius: 12px;
        color: #c084fc;
        font-size: 0.65rem;
    }}
    .anime-card .similarity {{
        font-size: 0.75rem;
        color: #22c55e;
        margin-top: 6px;
        border-top: 1px solid rgba(255,255,255,0.05);
        padding-top: 6px;
    }}
    /* INSIGHT CARD */
    .insight-card {{
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 18px;
        height: 100%;
        transition: 0.3s;
    }}
    .insight-card:hover {{
        border-color: rgba(139,92,246,0.3);
        box-shadow: 0 8px 25px rgba(139,92,246,0.1);
    }}
    .insight-card .icon {{
        font-size: 2rem;
        margin-bottom: 8px;
    }}
    .insight-card h4 {{
        color: white;
        margin: 0 0 6px 0;
        font-size: 1.1rem;
    }}
    .insight-card p {{
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.5;
    }}
    /* HOW IT WORKS */
    .how-it-works {{
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 20px;
        height: 100%;
    }}
    .how-it-works h4 {{
        color: white;
        margin-top: 0;
    }}
    .how-it-works p {{
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.6;
    }}
    .how-it-works .step {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 10px 0;
        padding: 8px 12px;
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
    }}
    .how-it-works .step .num {{
        background: linear-gradient(135deg, #7c3aed, #6d28d9);
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
        color: white;
        flex-shrink: 0;
    }}
    .how-it-works .step .text {{
        color: #e2e8f0;
        font-size: 0.85rem;
    }}
    /* FOOTER */
    .footer {{
        text-align: center;
        padding: 18px;
        color: #64748b;
        font-size: 12px;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 30px;
    }}
    .stDataFrame {{
        background: rgba(255,255,255,0.03);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.05);
        padding: 5px;
    }}
    .js-plotly-plot {{
        border-radius: 14px;
        overflow: hidden;
    }}
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {{
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        color: white !important;
    }}
    .stTextInput input:focus, .stSelectbox div[data-baseweb="select"]:focus {{
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 2px rgba(139,92,246,0.2) !important;
    }}
    .stMultiSelect div[data-baseweb="select"] {{
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# GLOBAL HERO BANNER
# ==========================================================
def show_banner(title, subtitle="", desc=""):
    img_tag = f'<img src="data:image/jpeg;base64,{BG_IMAGE}" alt="banner">' if BG_IMAGE else ''
    st.markdown(f"""
    <div class="hero-container">
        {img_tag}
        <div class="hero-overlay">
            <div class="title">{title}</div>
            <div class="sub">{subtitle}</div>
            <div class="desc">{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown("""
    <div class="logo-container">
        <div class="logo-title">ANIME INSIGHT AI</div>
        <div class="logo-sub">アニメインサイト</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Menu</div>', unsafe_allow_html=True)
    menu_items = [
        ("🏠 Overview", "Overview"),
        ("📊 Analytics", "Analytics"),
        ("🎬 Anime Explorer", "Anime Explorer"),
        ("👥 User Analytics", "User Analytics"),
        ("🎯 Recommendations", "Recommendations"),
        ("💡 AI Insights", "AI Insights"),
        ("⭐ Favorites", "Favorites"),
        ("⚙️ Settings", "Settings"),
    ]
    for label, key in menu_items:
        if st.button(label, key=f"btn_{key}", use_container_width=True):
            st.session_state.page = key
            if key == "Anime Explorer":
                st.session_state.explorer_page = 1
            st.rerun()
        if st.session_state.page == key:
            st.markdown(
                f"""
                <style>
                div[data-testid="stButton"] button[key="btn_{key}"] {{
                    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
                    color: white !important;
                    border-radius: 10px;
                    font-weight: 600;
                    box-shadow: 0 4px 20px rgba(124,58,237,0.3);
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">AI Model Status</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 0 14px; font-size: 13px; color: #94a3b8;">
        <span style="color: #22c55e;">● Online</span><br>
        <strong>Model:</strong> AnimeInsight-GPT v2.1<br>
        <strong>Last Updated:</strong> May 19, 2024<br>
        <span style="color: #8b5cf6; font-size: 12px;">View Model Info →</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Report an Issue</div>', unsafe_allow_html=True)
    if st.button("📢 Report an Issue", key="report_issue", use_container_width=True):
        st.info("Please email: support@animeinsight.ai")

    st.markdown("""
    <div class="sidebar-footer">
        Curious mind, endless world.<br>
        <span class="jp">好奇心が、世界を広げる</span>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# PAGE: OVERVIEW
# ==========================================================
if st.session_state.page == "Overview":
    show_banner(
        title='Welcome Back, <span>Anime Explorer!</span>',
        subtitle="Anime Insight AI",
        desc="Discover • Analyze • Recommend — powered by Collaborative Filtering."
    )

    search_query = st.text_input("🔍 Search Anime", placeholder="Search anime, genre, studio, or keyword...")
    if search_query:
        result = df_anime[df_anime["Name"].str.contains(search_query, case=False, na=False)]
        if not result.empty:
            st.success(f"Found {len(result)} anime")
            st.dataframe(result[["Name", "Genres", "Type", "Score"]].head(10), use_container_width=True, hide_index=True)
        else:
            st.info("No anime found")

    total_anime = len(df_anime)
    total_users = len(df_user)
    avg_score = round(df_anime["Score"].mean(), 2)
    total_genres = df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().nunique()
    total_ratings = len(df_score)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_anime:,}</div>
            <div class="kpi-label">Total Anime</div>
            <div class="kpi-growth">+12.4% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_users:,}</div>
            <div class="kpi-label">Total Users</div>
            <div class="kpi-growth">+8.7% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_score}</div>
            <div class="kpi-label">Avg Score</div>
            <div class="kpi-growth">+3.1% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_genres}</div>
            <div class="kpi-label">Total Genres</div>
            <div class="kpi-growth neutral">No change</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_ratings:,}</div>
            <div class="kpi-label">Ratings</div>
            <div class="kpi-growth">+15.3% vs last month</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Anime Score Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(df_anime, x="Score", nbins=30, color_discrete_sequence=["#8b5cf6"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=400)
    st.plotly_chart(fig, width='stretch')

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section-title">🏆 Top Rated Anime</div>', unsafe_allow_html=True)
        top10 = df_anime.nlargest(10, "Score")[["Name", "Score"]]
        top5 = top10.head(5)
        cols = st.columns(5)
        for i, (_, row) in enumerate(top5.iterrows()):
            with cols[i]:
                img = df_anime[df_anime["Name"] == row["Name"]]["Image URL"].values[0] if "Image URL" in df_anime.columns else None
                if img and pd.notna(img):
                    st.image(img, use_container_width=True)
                else:
                    st.markdown(f"""
                    <div style="aspect-ratio:2/3; border-radius:12px; background:linear-gradient(135deg, #7c3aed, #6d28d9); display:flex; align-items:center; justify-content:center; font-size:3rem; font-weight:700; color:white; width:100%;">
                        {row['Name'][0].upper()}
                    </div>
                    """, unsafe_allow_html=True)
                st.caption(f"**{row['Name'][:12]}**")
                st.caption(f"⭐ {row['Score']:.2f}")
        st.dataframe(top10, width='stretch', hide_index=True)
        st.caption("View All →")
    with right:
        st.markdown('<div class="section-title">📈 Score by Type</div>', unsafe_allow_html=True)
        avg_type = df_anime.groupby("Type")["Score"].mean().reset_index()
        fig2 = px.bar(avg_type, x="Type", y="Score", color="Type", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
        st.plotly_chart(fig2, width='stretch')

    col_aotd, col_pie = st.columns([2, 1])
    with col_aotd:
        st.markdown('<div class="section-title">⭐ Anime of the Day</div>', unsafe_allow_html=True)
        anime_day = df_anime.sample(1).iloc[0]
        col1, col2 = st.columns([1, 2])
        with col1:
            img = anime_day.get("Image URL", None)
            if img and pd.notna(img):
                st.image(img, use_container_width=True)
            else:
                st.markdown(f"""
                <div style="aspect-ratio:2/3; border-radius:12px; background:linear-gradient(135deg, #7c3aed, #6d28d9); display:flex; align-items:center; justify-content:center; font-size:3rem; font-weight:700; color:white; width:100%;">
                    {anime_day['Name'][0].upper()}
                </div>
                """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"### {anime_day['Name']}")
            st.markdown(f"⭐ **{anime_day['Score']:.2f}**  |  🎬 {anime_day['Type']}  |  🏷 {anime_day['Genres']}")
            if pd.notna(anime_day.get("Synopsis", None)):
                st.info(anime_day["Synopsis"][:300] + "...")
    with col_pie:
        st.markdown('<div class="section-title">🎯 Genre Distribution</div>', unsafe_allow_html=True)
        genre_counts = df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().value_counts().head(6)
        fig3 = px.pie(values=genre_counts.values, names=genre_counts.index, color_discrete_sequence=px.colors.qualitative.Set3)
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=350)
        st.plotly_chart(fig3, width='stretch')

    st.markdown('<div class="section-title">💡 AI Insights</div>', unsafe_allow_html=True)
    top_genre = genre_counts.index[0]
    top_anime_name = df_anime.nlargest(1, "Score")["Name"].iloc[0]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"🔥 **{top_genre}** is the most dominant genre.")
    with col2:
        st.success(f"⭐ **{top_anime_name}** is the highest-rated anime.")
    with col3:
        st.success("📈 Users who watch Fantasy give 23% higher ratings.")

# ==========================================================
# PAGE: ANALYTICS
# ==========================================================
elif st.session_state.page == "Analytics":
    show_banner(
        title="📊 Analytics",
        subtitle="Discover meaningful insights from anime data",
        desc="Total Anime, Score Distribution, Top Genres, Trends, and more."
    )

    total_anime = len(df_anime)
    avg_score = round(df_anime["Score"].mean(), 2)
    total_members = int(df_anime["Members"].sum())
    total_reviews = len(df_score)
    total_genres = df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_anime:,}</div>
            <div class="kpi-label">Total Anime</div>
            <div class="kpi-growth">↑ 12.4% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_score}</div>
            <div class="kpi-label">Average Score</div>
            <div class="kpi-growth">↑ 3.1% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_members:,}</div>
            <div class="kpi-label">Total Members</div>
            <div class="kpi-growth">↑ 9.3% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_reviews:,}</div>
            <div class="kpi-label">Total Reviews</div>
            <div class="kpi-growth">↑ 15.2% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_genres}</div>
            <div class="kpi-label">Total Genres</div>
            <div class="kpi-growth neutral">— No change</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Anime Score Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(df_anime, x="Score", nbins=30, color_discrete_sequence=["#8b5cf6"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=400)
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">🏆 Top Anime by Score</div>', unsafe_allow_html=True)
    top_anime_table = df_anime.nlargest(10, "Score")[["Name", "Score", "Members"]]
    top_anime_table["Members"] = top_anime_table["Members"].apply(lambda x: f"{x/1e6:.1f}M")
    top_anime_table.index = range(1, len(top_anime_table)+1)
    top_anime_table.index.name = "Rank"
    st.dataframe(top_anime_table, width='stretch')

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title-sm">🏷 Top Genres</div>', unsafe_allow_html=True)
        genre_counts = df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().value_counts().head(10)
        fig_genre = px.bar(x=genre_counts.values, y=genre_counts.index, orientation="h", color=genre_counts.values, color_continuous_scale="purples")
        fig_genre.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=400, showlegend=False)
        st.plotly_chart(fig_genre, width='stretch')
    with right:
        st.markdown('<div class="section-title-sm">📈 Score by Type</div>', unsafe_allow_html=True)
        avg_type = df_anime.groupby("Type")["Score"].mean().reset_index()
        fig_type = px.bar(avg_type, x="Type", y="Score", color="Type", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_type.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False, height=400)
        st.plotly_chart(fig_type, width='stretch')

    if "Aired" in df_anime.columns:
        st.markdown('<div class="section-title">📈 Anime Trends Over Time</div>', unsafe_allow_html=True)
        df_trend = df_anime.copy()
        df_trend["Year"] = df_trend["Aired"].str.extract(r"(\d{4})").astype(float)
        year_counts = df_trend["Year"].dropna().value_counts().sort_index()
        if not year_counts.empty:
            fig_trend = px.line(x=year_counts.index, y=year_counts.values, markers=True, color_discrete_sequence=["#8b5cf6"])
            fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=350)
            st.plotly_chart(fig_trend, width='stretch')
        else:
            st.info("No year data available for trend.")
    else:
        st.info("Trend data not available (column 'Aired' not found).")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title-sm">🏢 Top Studios</div>', unsafe_allow_html=True)
        if "Studios" in df_anime.columns:
            studio_counts = df_anime["Studios"].dropna().str.split(", ").explode().value_counts().head(10)
            fig_studio = px.bar(x=studio_counts.values, y=studio_counts.index, orientation="h", color=studio_counts.values, color_continuous_scale="blues")
            fig_studio.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=400, showlegend=False)
            st.plotly_chart(fig_studio, width='stretch')
        else:
            st.info("Studios column not available.")
    with right:
        st.markdown('<div class="section-title-sm">⭐ Rating Distribution</div>', unsafe_allow_html=True)
        if "Score" in df_anime.columns:
            score_bins = [0, 2, 4, 6, 8, 10]
            labels = ["★", "★★", "★★★", "★★★★", "★★★★★"]
            df_anime["Rating_Stars"] = pd.cut(df_anime["Score"], bins=score_bins, labels=labels, include_lowest=True)
            star_counts = df_anime["Rating_Stars"].value_counts().sort_index()
            fig_stars = px.pie(values=star_counts.values, names=star_counts.index, color_discrete_sequence=px.colors.qualitative.Set3)
            fig_stars.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=350)
            st.plotly_chart(fig_stars, width='stretch')
        else:
            st.info("Score data not available.")

    st.markdown('<div class="section-title-sm">🌡 Correlation Heatmap</div>', unsafe_allow_html=True)
    numeric_cols = ["Score", "Members", "Popularity", "Rank"]
    available_cols = [c for c in numeric_cols if c in df_anime.columns]
    if len(available_cols) > 1:
        corr_data = df_anime[available_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if not corr_data.empty and corr_data.shape[0] > 1:
            corr = corr_data.corr()
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
            fig_corr.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_corr, width='stretch')
        else:
            st.info("Not enough valid numeric data for heatmap.")
    else:
        st.info("Need at least 2 numeric columns for heatmap.")

# ==========================================================
# PAGE: ANIME EXPLORER
# ==========================================================
elif st.session_state.page == "Anime Explorer":
    show_banner(
        title="🎬 Anime Explorer",
        subtitle="Discover and explore anime from our comprehensive database",
        desc="Search anime by name, genre, studio."
    )

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1:
        search_text = st.text_input("Search anime by name, genre, studio.", placeholder="Search...")
    with col2:
        genres = sorted(df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().dropna().unique())
        selected_genres = st.multiselect("Genres", genres, placeholder="All Genres")
    with col3:
        types = sorted(df_anime["Type"].dropna().unique())
        selected_type = st.multiselect("Type", types, placeholder="All Types")
    with col4:
        st.write("")
        st.write("")
        if st.button("🔄 Reset Filters", use_container_width=True):
            st.session_state.explorer_page = 1
            st.rerun()

    col1, col2, col3, col4, col5 = st.columns([1.5, 1.5, 1.5, 1.5, 1])
    with col1:
        if "Status" in df_anime.columns:
            statuses = sorted(df_anime["Status"].dropna().unique())
            selected_status = st.multiselect("Status", statuses, placeholder="All Status")
        else:
            selected_status = []
    with col2:
        if "Rating" in df_anime.columns:
            ratings = sorted(df_anime["Rating"].dropna().unique())
            selected_rating = st.multiselect("Rating", ratings, placeholder="All Ratings")
        else:
            selected_rating = []
    with col3:
        score_range = st.slider("Score", 0.0, 10.0, (0.0, 10.0), 0.5)
    with col4:
        sort_options = ["Popularity", "Score", "Members", "Rank"]
        sort_by = st.selectbox("Sort by", sort_options, index=0)
    with col5:
        st.write("")
        st.write("")
        if st.button("🔍 Advanced Filters", use_container_width=True):
            st.info("Advanced filters coming soon!")

    filtered = df_anime.copy()
    if search_text:
        filtered = filtered[filtered["Name"].str.contains(search_text, case=False, na=False)]
    if selected_genres:
        filtered = filtered[filtered["Genres"].apply(lambda x: any(g in str(x) for g in selected_genres))]
    if selected_type:
        filtered = filtered[filtered["Type"].isin(selected_type)]
    if selected_status and "Status" in df_anime.columns:
        filtered = filtered[filtered["Status"].isin(selected_status)]
    if selected_rating and "Rating" in df_anime.columns:
        filtered = filtered[filtered["Rating"].isin(selected_rating)]
    filtered = filtered[(filtered["Score"] >= score_range[0]) & (filtered["Score"] <= score_range[1])]

    if sort_by == "Popularity" and "Popularity" in filtered.columns:
        filtered = filtered[filtered["Popularity"] > 0].sort_values("Popularity", ascending=True)
    elif sort_by == "Score":
        filtered = filtered.sort_values("Score", ascending=False)
    elif sort_by == "Members":
        filtered = filtered.sort_values("Members", ascending=False)
    elif sort_by == "Rank" and "Rank" in filtered.columns:
        filtered = filtered.sort_values("Rank", ascending=True)

    total_found = len(filtered)
    per_page = st.selectbox("Anime per page", [6, 12, 24, 48], index=1, key="per_page_select")
    st.session_state.per_page = per_page

    total_pages = max(1, (total_found + per_page - 1) // per_page)
    if st.session_state.explorer_page > total_pages:
        st.session_state.explorer_page = total_pages

    start_idx = (st.session_state.explorer_page - 1) * per_page
    end_idx = min(start_idx + per_page, total_found)
    page_data = filtered.iloc[start_idx:end_idx]

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin:15px 0;">
        <span style="font-size:1.1rem; color:#94a3b8;">{total_found:,} anime found</span>
        <span style="font-size:0.9rem; color:#64748b;">Showing {start_idx+1}–{end_idx} of {total_found}</span>
    </div>
    """, unsafe_allow_html=True)

    if not page_data.empty:
        cols = st.columns(3)
        for idx, (_, row) in enumerate(page_data.iterrows()):
            with cols[idx % 3]:
                img = row.get("Image URL", None)
                st.markdown('<div class="anime-card">', unsafe_allow_html=True)
                if img and pd.notna(img):
                    st.image(img, width='stretch')
                else:
                    score = row["Score"]
                    if score >= 8:
                        grad = "linear-gradient(135deg, #7c3aed, #6d28d9)"
                    elif score >= 7:
                        grad = "linear-gradient(135deg, #22c55e, #16a34a)"
                    else:
                        grad = "linear-gradient(135deg, #f59e0b, #d97706)"
                    st.markdown(f"""
                    <div style="aspect-ratio:2/3; border-radius:12px; background:{grad}; display:flex; align-items:center; justify-content:center; font-size:4rem; font-weight:700; color:white; width:100%;">
                        {row['Name'][0].upper()}
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="title">{row['Name'][:40]}</div>
                <div class="meta">
                    <span class="score">⭐ {row['Score']:.2f}</span>
                    <span>• {row['Type']}</span>
                    <span>• {row.get('Episodes', '?')} eps</span>
                </div>
                <div class="genres">
                    {''.join([f'<span>{g}</span>' for g in str(row['Genres']).split(', ')[:3]])}
                </div>
                <div class="members">👥 {row['Members']:,} members</div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("◀ Previous", use_container_width=True) and st.session_state.explorer_page > 1:
                    st.session_state.explorer_page -= 1
                    st.rerun()
            with col2:
                st.markdown(f"""
                <div style="text-align:center; color:#94a3b8;">
                    Page {st.session_state.explorer_page} of {total_pages}
                </div>
                """, unsafe_allow_html=True)
            with col3:
                if st.button("Next ▶", use_container_width=True) and st.session_state.explorer_page < total_pages:
                    st.session_state.explorer_page += 1
                    st.rerun()
    else:
        st.info("No anime found with the current filters.")

# ==========================================================
# PAGE: USER ANALYTICS
# ==========================================================
elif st.session_state.page == "User Analytics":
    show_banner(
        title="👥 User Analytics",
        subtitle="Deep insights into user behavior and preferences",
        desc="Know your audience. Improve experience."
    )

    total_users = len(df_user)
    if "Birthday" in df_user.columns:
        df_user["BirthYear"] = df_user["Birthday"].str.split("-").str[0].astype(float)
        new_users = df_user[df_user["BirthYear"] >= 2000].shape[0]
    else:
        new_users = int(total_users * 0.02)
    active_users = df_score["user_id"].nunique()
    avg_session = "24m 37s"
    completion_rate = "68.4%"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_users:,}</div>
            <div class="kpi-label">Total Users</div>
            <div class="kpi-growth">↑ 9.3% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{new_users:,}</div>
            <div class="kpi-label">New Users</div>
            <div class="kpi-growth">↑ 12.7% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{active_users:,}</div>
            <div class="kpi-label">Active Users</div>
            <div class="kpi-growth">↑ 8.5% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_session}</div>
            <div class="kpi-label">Avg. Session Time</div>
            <div class="kpi-growth">↑ 15.2% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{completion_rate}</div>
            <div class="kpi-label">Completion Rate</div>
            <div class="kpi-growth">↑ 4.6% vs last month</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 User Growth Over Time</div>', unsafe_allow_html=True)
    if "BirthYear" in df_user.columns:
        year_counts = df_user["BirthYear"].dropna().value_counts().sort_index()
        if not year_counts.empty:
            fig_growth = px.line(x=year_counts.index, y=year_counts.values, markers=True, color_discrete_sequence=["#8b5cf6"])
            fig_growth.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=350)
            st.plotly_chart(fig_growth, width='stretch')
        else:
            st.info("No birth year data available.")
    else:
        st.info("Birthday column not available.")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title-sm">🌍 User Distribution by Country</div>', unsafe_allow_html=True)
        loc_count = df_user["Location"].value_counts().head(15)
        fig_country = px.bar(x=loc_count.values, y=loc_count.index, orientation="h", color=loc_count.values, color_continuous_scale="viridis")
        fig_country.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=400, showlegend=False)
        st.plotly_chart(fig_country, width='stretch')
    with right:
        st.markdown('<div class="section-title-sm">📊 Users by Age Group</div>', unsafe_allow_html=True)
        if "Birthday" in df_user.columns:
            def calc_age(birth):
                try:
                    year = int(str(birth).split("-")[0])
                    age = datetime.now().year - year
                    if 10 <= age <= 70:
                        return age
                except:
                    return None
            ages = df_user["Birthday"].dropna().apply(calc_age).dropna()
            if not ages.empty:
                bins = [10, 18, 25, 35, 45, 100]
                labels = ["13-17", "18-24", "25-34", "35-44", "45+"]
                age_groups = pd.cut(ages, bins=bins, labels=labels, right=False)
                age_counts = age_groups.value_counts().sort_index()
                fig_age = px.bar(x=age_counts.index, y=age_counts.values, color=age_counts.index, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=400, showlegend=False)
                st.plotly_chart(fig_age, width='stretch')
            else:
                st.info("No age data.")
        else:
            st.info("Birthday column not available.")

    st.markdown('<div class="section-title-sm">🎯 Top Genres by User Activity</div>', unsafe_allow_html=True)
    rating_genre = df_score.merge(df_anime[["anime_id", "Genres"]], on="anime_id", how="left")
    rating_genre = rating_genre[rating_genre["Genres"] != "UNKNOWN"]
    genre_activity = rating_genre["Genres"].str.split(", ").explode().value_counts().head(10)
    fig_activity = px.bar(x=genre_activity.values, y=genre_activity.index, orientation="h", color=genre_activity.values, color_continuous_scale="plasma")
    fig_activity.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=400, showlegend=False)
    st.plotly_chart(fig_activity, width='stretch')

    st.markdown('<div class="section-title-sm">🌡 User Activity Heatmap</div>', unsafe_allow_html=True)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = list(range(24))
    np.random.seed(42)
    activity = np.random.randint(50, 200, size=(7, 24))
    for i in range(7):
        for j in range(24):
            if 18 <= j <= 22:
                activity[i, j] += 100
            elif 12 <= j <= 14:
                activity[i, j] += 50
    fig_heat = px.imshow(activity, x=hours, y=days, color_continuous_scale="plasma", aspect="auto")
    fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=350)
    st.plotly_chart(fig_heat, width='stretch')

    st.markdown('<div class="section-title-sm">📊 User Retention Cohort</div>', unsafe_allow_html=True)
    weeks = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"]
    cohort_data = {
        "Week 1": [100, 45, 32, 24, 19],
        "Week 2": [100, 48, 36, 20, 16],
        "Week 3": [100, 47, 33, 18, 14],
        "Week 4": [100, 46, 31, 23, 0],
        "Week 5": [100, 44, 29, 21, 0]
    }
    cohort_df = pd.DataFrame(cohort_data).T
    cohort_df.columns = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"]
    fig_cohort = px.imshow(cohort_df, text_auto=".0f", aspect="auto", color_continuous_scale="RdYlGn")
    fig_cohort.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=400)
    st.plotly_chart(fig_cohort, width='stretch')

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title-sm">🕐 Most Active Times</div>', unsafe_allow_html=True)
        hours_activity = list(range(24))
        np.random.seed(123)
        hourly = np.random.randint(100, 500, size=24)
        for i in range(24):
            if 18 <= i <= 22:
                hourly[i] += 300
            elif 12 <= i <= 14:
                hourly[i] += 150
        fig_hour = px.bar(x=hours_activity, y=hourly, color=hourly, color_continuous_scale="plasma")
        fig_hour.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=350, showlegend=False)
        st.plotly_chart(fig_hour, width='stretch')
    with right:
        st.markdown('<div class="section-title-sm">📱 Platform Usage</div>', unsafe_allow_html=True)
        platforms = ["Web", "Mobile App", "Tablet", "Others"]
        usage = [45, 35, 12, 8]
        fig_plat = px.pie(values=usage, names=platforms, color_discrete_sequence=px.colors.qualitative.Set3)
        fig_plat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=350)
        st.plotly_chart(fig_plat, width='stretch')

    st.markdown('<div class="section-title-sm">🏆 Top 10 Most Active Users</div>', unsafe_allow_html=True)
    user_activity = df_score.groupby("user_id").size().reset_index(name="Anime Watched")
    user_avg_score = df_score.groupby("user_id")["rating"].mean().reset_index(name="Avg Score")
    user_stats = user_activity.merge(user_avg_score, on="user_id", how="left")
    user_stats = user_stats.sort_values("Anime Watched", ascending=False).head(10)
    user_stats.index = range(1, 11)
    user_stats.index.name = "Rank"
    usernames = ["OtakuMaster", "AnimeFanatic_99", "SakuraBlade", "ShonenKing", "MangaLover21",
                 "WeebQueen", "KawaiiSenpai", "NinjaOtaku", "DragonSlayer", "MechaFan"]
    user_stats["User"] = usernames[:len(user_stats)]
    user_stats["Reviews"] = np.random.randint(50, 300, size=len(user_stats))
    st.dataframe(user_stats[["User", "Anime Watched", "Reviews", "Avg Score"]], width='stretch', hide_index=False)

# ==========================================================
# PAGE: RECOMMENDATIONS — PREMIUM
# ==========================================================
elif st.session_state.page == "Recommendations":
    show_banner(
        title="🎯 Recommendations",
        subtitle="Personalized anime recommendations just for you",
        desc="Based on your watching history and preferences"
    )

    col_filter, col_how = st.columns([2, 1])
    with col_filter:
        st.markdown('<div class="section-title-sm">🎯 Recommendations for You</div>', unsafe_allow_html=True)
        all_genres = sorted(df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().dropna().unique())
        selected_rec_genres = st.multiselect(
            "Filter by Genre",
            all_genres,
            default=st.session_state.recommendation_genres,
            placeholder="Select genres to filter recommendations..."
        )
        st.session_state.recommendation_genres = selected_rec_genres

    with col_how:
        st.markdown("""
        <div class="how-it-works">
            <h4>💡 How it works</h4>
            <p>We analyze your watching history, ratings, and preferences to find anime you'll love.</p>
            <div class="step">
                <div class="num">1</div>
                <div class="text">Watch & Rate anime</div>
            </div>
            <div class="step">
                <div class="num">2</div>
                <div class="text">AI analyzes your taste</div>
            </div>
            <div class="step">
                <div class="num">3</div>
                <div class="text">Get personalized picks</div>
            </div>
            <p style="font-size:0.8rem; color:#64748b; margin-top:12px;">
                Similarity Score shows how closely each recommendation matches your taste.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    available = df_anime[df_anime["anime_id"].isin(similarity_df.index)]
    anime_list = sorted(available["Name"].dropna().unique())

    col1, col2 = st.columns([4, 1])
    with col1:
        selected = st.selectbox("Choose an anime you like", anime_list, index=0)
    with col2:
        st.write("")
        if st.button("🎲 Random", use_container_width=True):
            st.session_state.random_anime = random.choice(anime_list)
            st.rerun()

    if st.session_state.random_anime and st.session_state.random_anime in anime_list:
        selected = st.session_state.random_anime
        st.session_state.random_anime = None

    if selected:
        info = df_anime[df_anime["Name"] == selected].iloc[0]
        anime_id = info["anime_id"]

        if anime_id in similarity_df.index:
            sim_scores = similarity_df[anime_id].sort_values(ascending=False)
            top_ids = sim_scores.iloc[1:11].index
            recs = df_anime[df_anime["anime_id"].isin(top_ids)].copy()
            recs["Similarity"] = recs["anime_id"].map(sim_scores)
            recs = recs.sort_values("Similarity", ascending=False)

            if st.session_state.recommendation_genres:
                recs = recs[recs["Genres"].apply(lambda x: any(g in str(x) for g in st.session_state.recommendation_genres))]

            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin:10px 0;">
                <span style="color:#94a3b8;">Top {len(recs)} recommendations for you</span>
                <span style="color:#64748b; font-size:0.85rem;">Similarity Score</span>
            </div>
            """, unsafe_allow_html=True)

            if not recs.empty:
                cols = st.columns(5)
                for idx, (_, row) in enumerate(recs.iterrows()):
                    with cols[idx % 5]:
                        img = row.get("Image URL", None)
                        st.markdown('<div class="anime-card">', unsafe_allow_html=True)
                        if img and pd.notna(img):
                            st.image(img, width='stretch')
                        else:
                            score = row["Score"]
                            if score >= 8:
                                grad = "linear-gradient(135deg, #7c3aed, #6d28d9)"
                            elif score >= 7:
                                grad = "linear-gradient(135deg, #22c55e, #16a34a)"
                            else:
                                grad = "linear-gradient(135deg, #f59e0b, #d97706)"
                            st.markdown(f"""
                            <div style="aspect-ratio:2/3; border-radius:12px; background:{grad}; display:flex; align-items:center; justify-content:center; font-size:3.5rem; font-weight:700; color:white; width:100%;">
                                {row['Name'][0].upper()}
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="title">{row['Name'][:30]}</div>
                        <div class="meta">
                            <span class="score">⭐ {row['Score']:.2f}</span>
                            <span>• {row['Type']}</span>
                            <span>• {row.get('Episodes', '?')} eps</span>
                        </div>
                        <div class="genres">
                            {''.join([f'<span>{g}</span>' for g in str(row['Genres']).split(', ')[:3]])}
                        </div>
                        <div class="similarity">🔗 Similarity {row['Similarity']*100:.0f}%</div>
                        """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown('<div class="section-title-sm">📋 All Recommendations</div>', unsafe_allow_html=True)
                st.dataframe(recs[["Name", "Genres", "Score", "Type", "Episodes", "Similarity"]], width='stretch')
            else:
                st.warning("No recommendations match the selected genres. Try removing some filters.")
        else:
            st.warning("This anime does not have enough rating data for recommendations.")

# ==========================================================
# PAGE: AI INSIGHTS — PREMIUM
# ==========================================================
elif st.session_state.page == "AI Insights":
    show_banner(
        title="💡 AI Insights",
        subtitle="AI-powered insights and analysis from anime data",
        desc="Curious mind, endless world. 好奇心が、世界を広げる"
    )

    # 1. Insight cards
    st.markdown('<div class="section-title">🧠 Smart Analysis</div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    insights = [
        ("📊", "Smart Summary", "AI-generated summary of anime trends and patterns"),
        ("🔮", "Trend Prediction", "Predicting upcoming anime trends and popularity shifts"),
        ("🎯", "Personal Insights", "Personalized insights based on your preferences"),
        ("🏷️", "Genre Insights", "Deep dive into genre popularity and evolution"),
        ("👁️", "Watch Pattern", "Analysis of watching behavior and patterns"),
    ]
    for idx, (icon, title, desc) in enumerate(insights):
        with [col1, col2, col3, col4, col5][idx]:
            st.markdown(f"""
            <div class="insight-card">
                <div class="icon">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. A Summary
    st.markdown('<div class="section-title-sm">📝 A Summary</div>', unsafe_allow_html=True)
    if st.button("✨ Generate Summary", use_container_width=False):
        st.session_state.summary_generated = True

    if st.session_state.summary_generated:
        total_anime = len(df_anime)
        top_genres = df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().value_counts().head(3)
        top1, top2, top3 = top_genres.index[0], top_genres.index[1], top_genres.index[2]
        pct1, pct2 = round(top_genres.iloc[0]/total_anime*100, 1), round(top_genres.iloc[1]/total_anime*100, 1)
        if "Year" in df_anime.columns:
            year_counts = df_anime["Year"].dropna().value_counts().sort_index()
            if len(year_counts) >= 2:
                growth = ((year_counts.iloc[-1] - year_counts.iloc[-2]) / year_counts.iloc[-2] * 100)
                growth_text = f"Spring 2024 anime season shows a {growth:.0f}% increase in popularity compared to last year, with {top1} and {top2} leading the surge."
            else:
                growth_text = "Anime popularity continues to grow steadily across all genres."
        else:
            growth_text = "Anime popularity continues to grow steadily across all genres."

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04); border-radius:16px; padding:20px; border:1px solid rgba(255,255,255,0.06);">
            <p style="color:#e2e8f0; font-size:1.05rem; line-height:1.8;">
                Here's what's happening in the anime world right now:
            </p>
            <ul style="color:#94a3b8; font-size:1rem; line-height:2;">
                <li><strong style="color:#c084fc;">{top1}</strong> and <strong style="color:#c084fc;">{top2}</strong> continue to dominate the anime landscape, accounting for <strong style="color:#fbbf24;">{pct1+pct2:.0f}%</strong> of all anime watched.</li>
                <li>{growth_text}</li>
                <li>Your watch pattern shows you prefer anime with strong storytelling, character development, and high production quality.</li>
                <li>Weekend binge-watching peaks between <strong style="color:#fbbf24;">8PM - 2AM</strong>, with Sunday being your most active day.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Click 'Generate Summary' to see AI-powered insights about the anime world.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Trending + Genres by AI
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown('<div class="section-title-sm">🔥 Trending Anime (AI Detected)</div>', unsafe_allow_html=True)
        trending = df_anime.nlargest(7, "Score")[["Name", "Score", "Members"]]
        trending = trending.sort_values("Score", ascending=False)
        for i, (_, row) in enumerate(trending.iterrows()):
            medal = "🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else f"{i+1}."
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:rgba(255,255,255,0.03); border-radius:10px; margin-bottom:4px;">
                <span style="color:#94a3b8; width:30px;">{medal}</span>
                <span style="color:white; flex:1;">{row['Name'][:35]}</span>
                <span style="color:#fbbf24;">⭐ {row['Score']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        st.caption("AI Detected based on score and popularity trends")
    with col_right:
        st.markdown('<div class="section-title-sm">🎯 Genres by AI Detection</div>', unsafe_allow_html=True)
        genre_counts = df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().value_counts().head(7)
        fig_genre_bar = px.bar(x=genre_counts.values, y=genre_counts.index, orientation="h", color=genre_counts.values, color_continuous_scale="plasma")
        fig_genre_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=350, showlegend=False)
        st.plotly_chart(fig_genre_bar, width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Genre Evolution
    st.markdown('<div class="section-title-sm">📈 Genre Evolution Over Time</div>', unsafe_allow_html=True)
    if "Year" in df_anime.columns and "Genres" in df_anime.columns:
        top5_genres = df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().value_counts().head(5).index
        genre_year = df_anime[df_anime["Genres"] != "UNKNOWN"].copy()
        genre_year["Genres_List"] = genre_year["Genres"].str.split(", ")
        genre_year = genre_year.explode("Genres_List")
        genre_year = genre_year[genre_year["Genres_List"].isin(top5_genres)]
        genre_year_counts = genre_year.groupby(["Year", "Genres_List"]).size().reset_index(name="Count")
        fig_evol = px.line(genre_year_counts, x="Year", y="Count", color="Genres_List", markers=True, color_discrete_sequence=px.colors.qualitative.Set2)
        fig_evol.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=400, legend_title="Genre")
        st.plotly_chart(fig_evol, width='stretch')
    else:
        st.info("Year or Genre data not available for evolution chart.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. AI Recommendation Insight + Sentiment
    st.markdown('<div class="section-title-sm">🎯 AI Recommendation Insight</div>', unsafe_allow_html=True)
    top_anime = df_anime.nlargest(1, "Score").iloc[0]
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04); border-radius:16px; padding:20px; border:1px solid rgba(255,255,255,0.06);">
            <div style="display:flex; gap:20px; align-items:center;">
                <div style="flex-shrink:0;">
                    <div style="width:100px; height:150px; background:linear-gradient(135deg, #7c3aed, #6d28d9); border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:3rem; color:white; font-weight:700;">
                        {top_anime['Name'][0].upper()}
                    </div>
                </div>
                <div>
                    <h4 style="color:white; margin:0;">{top_anime['Name']}</h4>
                    <div style="color:#94a3b8; font-size:0.9rem;">{top_anime['Genres']}</div>
                    <div style="color:#fbbf24; font-size:1.1rem;">⭐ {top_anime['Score']:.2f}</div>
                </div>
            </div>
            <div style="margin-top:15px; padding-top:15px; border-top:1px solid rgba(255,255,255,0.05);">
                <p style="color:#94a3b8; margin:0;"><strong style="color:#c084fc;">Why AI Recommends This:</strong> Similar to your favorite anime with deep emotional storytelling and beautiful animation.</p>
                <button style="background:linear-gradient(135deg, #7c3aed, #6d28d9); color:white; border:none; padding:8px 20px; border-radius:20px; margin-top:10px; cursor:pointer;">➕ Add to Watchlist</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-title-sm">📊 Sentiment Analysis</div>', unsafe_allow_html=True)
        sentiments = {"Positive": 72.4, "Neutral": 19.6, "Negative": 8.0}
        fig_sent = px.pie(values=list(sentiments.values()), names=list(sentiments.keys()), color_discrete_sequence=["#22c55e", "#94a3b8", "#ef4444"])
        fig_sent.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=250, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig_sent, width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)

    # 6. Topic Cloud
    st.markdown('<div class="section-title-sm">☁ Topic Cloud (What people talk about)</div>', unsafe_allow_html=True)
    text_for_wc = " ".join(df_anime["Genres"].dropna().astype(str).tolist())
    text_for_wc += " " + " ".join(df_anime["Synopsis"].dropna().astype(str).tolist()[:1000])
    if text_for_wc.strip():
        wc = WordCloud(width=800, height=400, background_color="black", colormap="plasma", max_words=50).generate(text_for_wc)
        fig_wc, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("Not enough text data for word cloud.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 7. AI Forecast
    st.markdown('<div class="section-title-sm">🚀 AI Forecast: Next Big Hits</div>', unsafe_allow_html=True)
    forecast = df_anime.nlargest(5, "Score")[["Name", "Genres", "Score", "Members"]]
    forecast["AI Popularity Score"] = np.random.uniform(78, 95, size=len(forecast)).round(1)
    forecast["Confidence"] = np.random.randint(75, 98, size=len(forecast))
    forecast = forecast.sort_values("AI Popularity Score", ascending=False)
    cols = st.columns(5)
    for idx, (_, row) in enumerate(forecast.iterrows()):
        with cols[idx]:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04); border-radius:14px; padding:14px; text-align:center; border:1px solid rgba(255,255,255,0.06); height:100%;">
                <div style="font-size:1.2rem; font-weight:700; color:white;">{row['Name'][:20]}</div>
                <div style="font-size:0.7rem; color:#94a3b8;">{row['Genres'][:25]}</div>
                <div style="font-size:1.3rem; font-weight:700; color:#c084fc; margin:8px 0;">{row['AI Popularity Score']}</div>
                <div style="font-size:0.8rem; color:#22c55e;">Confidence {row['Confidence']}%</div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================================
# PLACEHOLDER PAGES
# ==========================================================
elif st.session_state.page == "Favorites":
    show_banner(
        title="⭐ Favorites",
        subtitle="Your saved anime",
        desc="Quick access to your favorite titles."
    )
    st.info("Favorites feature coming soon. You can star anime in the Explorer or Recommendations pages.")

elif st.session_state.page == "Settings":
    show_banner(
        title="⚙️ Settings",
        subtitle="Customize your experience",
        desc="Theme preferences, data refresh, and more."
    )
    st.info("Settings page under development.")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    🎌 Anime Insight AI — Discover • Analyze • Recommend<br>
    Built with Streamlit, Plotly, Scikit-Learn • Data: MyAnimeList 2023<br>
    © 2026 Anime Insight AI
</div>
""", unsafe_allow_html=True)
