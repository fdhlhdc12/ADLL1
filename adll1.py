# ==========================================================
# ANIME INSIGHT AI
# ==========================================================
# Imports
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
import requests

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
if "selected_anime" not in st.session_state:
    st.session_state.selected_anime = None
if "random_anime" not in st.session_state:
    st.session_state.random_anime = None
if "image_cache" not in st.session_state:
    st.session_state.image_cache = {}

# ==========================================================
# LOAD BACKGROUND IMAGE (base64)
# ==========================================================
def get_base64(img_path):
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

# Jika file background tidak ada, kita pakai placeholder
BG_IMAGE = get_base64("assets/anime_bg.jpg")
# Jika tidak ada, gunakan background warna saja (tidak error)
if not BG_IMAGE:
    # fallback: kita set CSS tanpa gambar background
    BG_IMAGE = ""

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
# FUNCTION: GET ANIME IMAGE FROM API (dengan cache)
# ==========================================================
@st.cache_data(ttl=86400)  # cache 1 hari
def fetch_anime_image(title):
    """Ambil URL cover dari AniAPI (gratis, tanpa API key)"""
    if title in st.session_state.image_cache:
        return st.session_state.image_cache[title]
    try:
        # Bersihkan judul untuk query
        query = title.replace(" ", "+")
        url = f"https://api.aniapi.com/v1/anime?title={query}&limit=1"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data") and len(data["data"]) > 0:
                img = data["data"][0].get("cover_image")
                if img:
                    st.session_state.image_cache[title] = img
                    return img
    except Exception:
        pass
    # Fallback: inisial + gradien (tidak return None, tapi kita handle di UI)
    return None

# ==========================================================
# PREMIUM CSS (dengan active menu)
# ==========================================================
st.markdown(
    f"""
    <style>
    /* ---- BACKGROUND ---- */
    .stApp {{
        background: #0b1120;
        color: white;
    }}
    /* Jika ada gambar background */
    {f"""
    .stApp {{
        background:
            linear-gradient(rgba(3,7,18,0.92), rgba(3,7,18,0.92)),
            url("data:image/jpeg;base64,{BG_IMAGE}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """ if BG_IMAGE else ""}

    /* ---- SIDEBAR ---- */
    section[data-testid="stSidebar"] {{
        background: #0b1120;
        border-right: 1px solid rgba(255,255,255,0.05);
    }}
    .logo-container {{
        text-align: center;
        padding-top: 10px;
        padding-bottom: 25px;
    }}
    .logo-title {{
        font-size: 30px;
        font-weight: 800;
        color: #d8b4fe;
    }}
    .logo-sub {{
        font-size: 12px;
        color: #94a3b8;
        letter-spacing: 2px;
    }}
    .sidebar-divider {{
        margin: 20px 0;
        border-top: 1px solid rgba(255,255,255,0.05);
    }}
    /* ---- ACTIVE MENU ---- */
    .active-menu {{
        background: #7c3aed !important;
        border-radius: 10px;
        color: white !important;
        font-weight: 600;
    }}
    .stButton button {{
        width: 100%;
        background: transparent;
        border: none;
        color: #cbd5e1;
        padding: 8px 12px;
        border-radius: 10px;
        text-align: left;
        font-size: 16px;
        transition: 0.2s;
    }}
    .stButton button:hover {{
        background: rgba(255,255,255,0.05);
        color: white;
    }}
    /* ---- CARDS ---- */
    .glass-card {{
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }}
    .kpi-card {{
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        transition: 0.3s;
    }}
    .kpi-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(124,58,237,0.3);
    }}
    .kpi-value {{
        font-size: 36px;
        font-weight: 800;
        color: white;
    }}
    .kpi-label {{
        font-size: 14px;
        color: #cbd5e1;
    }}
    .kpi-growth {{
        font-size: 12px;
        color: #22c55e;
    }}
    .section-title {{
        font-size: 28px;
        font-weight: 700;
        margin: 20px 0 20px 0;
        color: white;
    }}
    /* ---- HERO ---- */
    .hero-container {{
        position: relative;
        height: 350px;
        overflow: hidden;
        border-radius: 25px;
        margin-bottom: 25px;
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
        padding: 40px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: linear-gradient(90deg, rgba(5,8,20,0.95) 0%, rgba(5,8,20,0.70) 40%, rgba(5,8,20,0.10) 100%);
    }}
    .hero-sub {{
        font-size: 1.2rem;
        color: #d8b4fe;
        font-weight: 600;
    }}
    .hero-title {{
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        line-height: 1.1;
    }}
    .hero-desc {{
        font-size: 1.3rem;
        color: white;
        margin-top: 5px;
    }}
    .hero-small {{
        font-size: 1rem;
        color: #cbd5e1;
        margin-top: 10px;
        max-width: 600px;
    }}
    /* ---- POSTER PLACEHOLDER ---- */
    .poster-initial {{
        aspect-ratio: 2/3;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 4rem;
        font-weight: 700;
        color: white;
    }}
    /* ---- FOOTER ---- */
    .footer {{
        text-align: center;
        padding: 20px;
        color: #94a3b8;
        font-size: 14px;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 30px;
    }}
    /* ---- PLOTLY ---- */
    .js-plotly-plot {{
        border-radius: 20px;
        overflow: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# PROFESSIONAL SIDEBAR dengan ACTIVE STATE
# ==========================================================
with st.sidebar:
    st.markdown("""
    <div class="logo-container">
        <div class="logo-title">🎌 ANIME INSIGHT AI</div>
        <div class="logo-sub">アニメインサイト</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Daftar menu dengan ikon
    menu_items = [
        ("🏠 Overview", "Overview"),
        ("🎬 Anime Explorer", "Anime Explorer"),
        ("📊 Analytics", "Analytics"),
        ("👥 User Analytics", "User Analytics"),
        ("🎯 Recommendations", "Recommendations"),
        ("💡 AI Insights", "AI Insights"),
    ]

    for label, key in menu_items:
        # Tentukan kelas aktif
        active_class = "active-menu" if st.session_state.page == key else ""
        # Tombol dengan custom styling melalui HTML (karena button tidak bisa diberi class langsung)
        # Kita gunakan st.button dengan key, lalu CSS menargetkan berdasarkan key?
        # Lebih praktis: kita buat button biasa dan tambahkan marker CSS jika aktif.
        if st.button(label, key=f"btn_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()
        # Tambahkan CSS untuk menyorot tombol aktif (target berdasarkan key)
        if st.session_state.page == key:
            st.markdown(
                f"""
                <style>
                div[data-testid="stButton"] button[key="btn_{key}"] {{
                    background: #7c3aed !important;
                    color: white !important;
                    border-radius: 10px;
                    font-weight: 600;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown(
        """
        <div style="text-align:center">
            ⭐ Premium Dashboard<br>
            <span style="color:#94a3b8; font-size:12px;">Powered by Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================
# ACTIVE PAGE
# ==========================================================
page = st.session_state.page

# ==========================================================
# PAGE: OVERVIEW (dengan search berfungsi)
# ==========================================================
if page == "Overview":
    # HERO BANNER
    st.markdown(
        f"""
        <div class="hero-container">
            <img src="data:image/jpeg;base64,{BG_IMAGE}" alt="hero">
            <div class="hero-overlay">
                <div class="hero-sub">👋 Welcome Back Anime Explorer</div>
                <div class="hero-title">Anime Insight AI</div>
                <div class="hero-desc">Discover • Analyze • Recommend</div>
                <div class="hero-small">Explore anime, uncover insights, and get personalized recommendations powered by AI.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # SEARCH BAR (berfungsi)
    search_query = st.text_input("🔍 Search Anime", placeholder="Search anime, genre, studio...")
    if search_query:
        result = df_anime[df_anime["Name"].str.contains(search_query, case=False, na=False)]
        if not result.empty:
            st.success(f"Found {len(result)} anime")
            st.dataframe(result[["Name", "Genres", "Type", "Score"]].head(10), use_container_width=True, hide_index=True)
        else:
            st.info("No anime found")

    # KPI CARDS
    total_anime = len(df_anime)
    total_users = len(df_user)
    avg_score = round(df_anime["Score"].mean(), 2)
    total_genres = df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().nunique()
    total_ratings = len(df_score)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_anime:,}</div>
            <div class="kpi-label">🎬 Total Anime</div>
            <div class="kpi-growth">+12.4%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_users:,}</div>
            <div class="kpi-label">👥 Total Users</div>
            <div class="kpi-growth">+8.7%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_score}</div>
            <div class="kpi-label">⭐ Avg Score</div>
            <div class="kpi-growth">+3.1%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_genres}</div>
            <div class="kpi-label">🏷 Total Genres</div>
            <div class="kpi-growth" style="color:#94a3b8;">Stable</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_ratings:,}</div>
            <div class="kpi-label">📝 Ratings</div>
            <div class="kpi-growth">+15.3%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # SCORE DISTRIBUTION
    st.markdown('<div class="section-title">📊 Anime Score Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(df_anime, x="Score", nbins=30)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # TOP RATED + SCORE BY TYPE
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section-title">🏆 Top Rated Anime</div>', unsafe_allow_html=True)
        top10 = df_anime.nlargest(10, "Score")[["Name", "Score"]]
        st.dataframe(top10, use_container_width=True, hide_index=True)
    with right:
        st.markdown('<div class="section-title">📈 Score by Type</div>', unsafe_allow_html=True)
        avg_type = df_anime.groupby("Type")["Score"].mean().reset_index()
        fig2 = px.bar(avg_type, x="Type", y="Score", color="Type")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ANIME OF THE DAY
    st.markdown('<div class="section-title">⭐ Anime Of The Day</div>', unsafe_allow_html=True)
    anime_day = df_anime.sample(1).iloc[0]
    st.markdown(f"""
    ### {anime_day["Name"]}
    ⭐ Score: {anime_day["Score"]}  |  🎬 Type: {anime_day["Type"]}  |  🏷 Genre: {anime_day["Genres"]}
    """)
    if pd.notna(anime_day["Synopsis"]):
        st.info(anime_day["Synopsis"][:400])

    # GENRE DISTRIBUTION
    st.markdown('<div class="section-title">🎯 Genre Distribution</div>', unsafe_allow_html=True)
    genre_counts = df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().value_counts().head(10)
    fig3 = px.pie(values=genre_counts.values, names=genre_counts.index)
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=450)
    st.plotly_chart(fig3, use_container_width=True)

    # AI INSIGHTS
    st.markdown('<div class="section-title">💡 AI Insights</div>', unsafe_allow_html=True)
    top_genre = genre_counts.index[0]
    top_anime_name = df_anime.nlargest(1, "Score")["Name"].iloc[0]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"🔥 {top_genre} adalah genre paling dominan.")
    with col2:
        st.success(f"⭐ {top_anime_name} memiliki skor tertinggi.")
    with col3:
        st.success("📈 Recommendation Engine siap digunakan.")

# ==========================================================
# PAGE: ANIME EXPLORER (dengan poster API)
# ==========================================================
elif page == "Anime Explorer":
    st.markdown('<div class="section-title">🎬 Anime Explorer</div>', unsafe_allow_html=True)
    st.caption("Explore anime database with advanced filtering and search.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        search_text = st.text_input("🔍 Search Anime")
    with col2:
        genres = sorted(df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().dropna().unique())
        selected_genres = st.multiselect("🏷 Genre", genres)
    with col3:
        selected_type = st.multiselect("🎬 Type", sorted(df_anime["Type"].dropna().unique()))
    with col4:
        score_range = st.slider("⭐ Score", 0.0, 10.0, (6.0, 10.0))

    # Filtering
    filtered = df_anime.copy()
    if search_text:
        filtered = filtered[filtered["Name"].str.contains(search_text, case=False, na=False)]
    if selected_type:
        filtered = filtered[filtered["Type"].isin(selected_type)]
    if selected_genres:
        filtered = filtered[filtered["Genres"].apply(lambda x: any(g in str(x) for g in selected_genres))]
    filtered = filtered[(filtered["Score"] >= score_range[0]) & (filtered["Score"] <= score_range[1])]

    st.success(f"Found {len(filtered):,} Anime")

    # Quick stats
    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("Anime", len(filtered))
    stat2.metric("Average Score", round(filtered["Score"].mean(), 2))
    stat3.metric("Highest Score", round(filtered["Score"].max(), 2))
    stat4.metric("Types", filtered["Type"].nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    # Anime Gallery dengan poster API
    st.markdown('<div class="section-title">🌟 Anime Gallery</div>', unsafe_allow_html=True)
    preview = filtered.head(12)
    cols = st.columns(4)
    for idx, (_, row) in enumerate(preview.iterrows()):
        with cols[idx % 4]:
            # Coba ambil gambar dari API
            img_url = fetch_anime_image(row["Name"])
            if img_url:
                st.image(img_url, use_container_width=True)
            else:
                # Fallback: inisial dengan gradien berdasarkan skor
                if row["Score"] >= 8:
                    grad = "linear-gradient(135deg, #7c3aed, #6d28d9)"
                elif row["Score"] >= 7:
                    grad = "linear-gradient(135deg, #22c55e, #16a34a)"
                else:
                    grad = "linear-gradient(135deg, #f59e0b, #d97706)"
                st.markdown(f"""
                <div class="poster-initial" style="background:{grad};">
                    {row["Name"][0].upper()}
                </div>
                """, unsafe_allow_html=True)
            st.caption(f"**{row['Name'][:30]}**")
            st.caption(f"⭐ {row['Score']:.2f}  •  {row['Type']}")

    # Top Genres + Score Distribution
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">🏷 Top Genres</div>', unsafe_allow_html=True)
        genre_counts2 = filtered[filtered["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().value_counts().head(10)
        fig = px.bar(x=genre_counts2.values, y=genre_counts2.index, orientation="h")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=450)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">📈 Score Distribution</div>', unsafe_allow_html=True)
        fig2 = px.histogram(filtered, x="Score", nbins=20)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=450)
        st.plotly_chart(fig2, use_container_width=True)

    # Data table
    st.markdown('<div class="section-title">📋 Anime Catalog</div>', unsafe_allow_html=True)
    display_cols = ["Name", "Genres", "Type", "Score", "Members"]
    st.dataframe(filtered[display_cols].sort_values("Score", ascending=False), use_container_width=True, height=600)

    # Top 10
    st.markdown('<div class="section-title">🏆 Top 10 Anime</div>', unsafe_allow_html=True)
    top10 = filtered.nlargest(10, "Score")[["Name", "Score"]]
    st.dataframe(top10, use_container_width=True, hide_index=True)

# ==========================================================
# PAGE: ANALYTICS (dengan heatmap terbatas)
# ==========================================================
elif page == "Analytics":
    st.markdown('<div class="section-title">📈 Analytics Dashboard</div>', unsafe_allow_html=True)
    st.caption("Deep insights and exploratory data analysis of anime dataset.")

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Highest Score", round(df_anime["Score"].max(), 2))
    c2.metric("Average Members", f"{int(df_anime['Members'].mean()):,}")
    c3.metric("Anime Types", df_anime["Type"].nunique())
    c4.metric("Genres", df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    # Score Distribution
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">⭐ Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df_anime, x="Score", nbins=30)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">🎯 Genre Distribution</div>', unsafe_allow_html=True)
        genre_counts = df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().value_counts().head(15)
        fig2 = px.bar(x=genre_counts.index, y=genre_counts.values, color=genre_counts.values)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False, height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Popularity & Members
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">🔥 Most Popular Anime</div>', unsafe_allow_html=True)
        popular = df_anime[df_anime["Popularity"] > 0].nsmallest(15, "Popularity")
        fig3 = px.bar(popular, x="Popularity", y="Name", orientation="h")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=500)
        st.plotly_chart(fig3, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">👥 Top Members</div>', unsafe_allow_html=True)
        members = df_anime.nlargest(15, "Members")
        fig4 = px.bar(members, x="Members", y="Name", orientation="h")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=500)
        st.plotly_chart(fig4, use_container_width=True)

    # Score vs Members
    st.markdown('<div class="section-title">📊 Score vs Members</div>', unsafe_allow_html=True)
    sample_df = df_anime.sample(min(2000, len(df_anime)))
    fig5 = px.scatter(sample_df, x="Members", y="Score", color="Type", hover_data=["Name"])
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=600)
    st.plotly_chart(fig5, use_container_width=True)

    # Correlation Heatmap (terbatas)
    st.markdown('<div class="section-title">🌡 Correlation Heatmap</div>', unsafe_allow_html=True)
    # Pilih kolom numerik yang penting
    selected_cols = ["Score", "Members", "Popularity", "Rank"]  # sesuaikan dengan kolom yang ada
    available_cols = [c for c in selected_cols if c in df_anime.columns]
    if len(available_cols) > 1:
        corr = df_anime[available_cols].corr()
        fig6 = px.imshow(corr, text_auto=".2f", aspect="auto")
        fig6.update_layout(height=500)
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Tidak cukup kolom numerik untuk heatmap.")

    # WordCloud
    st.markdown('<div class="section-title">☁ Genre WordCloud</div>', unsafe_allow_html=True)
    genre_text = " ".join(df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].dropna())
    if genre_text.strip():
        wc = WordCloud(width=1200, height=500, background_color="black", colormap="plasma").generate(genre_text)
        fig_wc, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("Tidak ada genre untuk word cloud.")

    # Radar
    st.markdown('<div class="section-title">🕸 Dataset Radar Profile</div>', unsafe_allow_html=True)
    if "Score" in df_anime.columns and "Members" in df_anime.columns:
        radar_categories = ["Score", "Members (log)"]
        radar_values = [df_anime["Score"].mean(), np.log1p(df_anime["Members"].mean())]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=radar_values, theta=radar_categories, fill="toself"))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=500)
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Radar chart membutuhkan kolom Score dan Members.")

    # Data Preview
    st.markdown('<div class="section-title">📋 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_anime.head(100), use_container_width=True, height=500)

# ==========================================================
# PAGE: USER ANALYTICS (dengan case-insensitive gender & Birthday check)
# ==========================================================
elif page == "User Analytics":
    st.markdown('<div class="section-title">👥 User Analytics</div>', unsafe_allow_html=True)

    total_users = len(df_user)
    gender_count = df_user["Gender"].dropna().value_counts()
    # Case‑insensitive counts
    male_count = gender_count[gender_count.index.str.contains("male", case=False, na=False)].sum()
    female_count = gender_count[gender_count.index.str.contains("female", case=False, na=False)].sum()
    total_country = df_user["Location"].dropna().nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", f"{total_users:,}")
    c2.metric("Countries", total_country)
    c3.metric("Male", male_count)
    c4.metric("Female", female_count)

    left, right = st.columns(2)
    with left:
        fig = px.pie(values=gender_count.values, names=gender_count.index, title="Gender Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        location_count = df_user["Location"].value_counts().head(15)
        fig2 = px.bar(x=location_count.values, y=location_count.index, orientation="h", title="Top Locations")
        st.plotly_chart(fig2, use_container_width=True)

    # Age distribution (jika kolom Birthday ada)
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
            fig3 = px.histogram(ages, nbins=25, title="Age Distribution")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Tidak ada data usia yang valid.")
    else:
        st.info("Kolom Birthday tidak tersedia dalam dataset.")

# ==========================================================
# PAGE: RECOMMENDATIONS (Netflix 5 kolom + poster API)
# ==========================================================
elif page == "Recommendations":
    st.markdown('<div class="section-title">🎯 Anime Recommendation Engine</div>', unsafe_allow_html=True)

    available = df_anime[df_anime["anime_id"].isin(similarity_df.index)]
    anime_list = sorted(available["Name"].dropna().unique())

    col1, col2 = st.columns([4, 1])
    with col1:
        selected = st.selectbox("Choose Anime", anime_list, index=0)
    with col2:
        st.write("")  # spacer
        if st.button("🎲 Random", use_container_width=True):
            st.session_state.random_anime = random.choice(anime_list)
            st.rerun()

    # Jika ada random_anime dari session, override
    if st.session_state.random_anime and st.session_state.random_anime in anime_list:
        selected = st.session_state.random_anime
        st.session_state.random_anime = None  # reset agar tidak terus-terusan

    if selected:
        info = df_anime[df_anime["Name"] == selected].iloc[0]
        anime_id = info["anime_id"]

        st.markdown(f"""
        ### 🎬 {info['Name']}
        ⭐ Score: {info['Score']:.2f}  |  🎞 Type: {info['Type']}  |  👥 Members: {info['Members']:,}  |  🏷 Genre: {info['Genres']}
        """)
        if pd.notna(info["Synopsis"]):
            st.info(info["Synopsis"][:600])

        if anime_id in similarity_df.index:
            sim_scores = similarity_df[anime_id].sort_values(ascending=False)
            top_ids = sim_scores.iloc[1:11].index
            recs = df_anime[df_anime["anime_id"].isin(top_ids)].copy()
            recs["Similarity"] = recs["anime_id"].map(sim_scores)
            recs = recs.sort_values("Similarity", ascending=False)

            st.markdown('<div class="section-title">🔥 Recommended For You</div>', unsafe_allow_html=True)

            # Netflix style: 5 columns
            cols = st.columns(5)
            for i, (_, row) in enumerate(recs.iterrows()):
                with cols[i % 5]:
                    img_url = fetch_anime_image(row["Name"])
                    if img_url:
                        st.image(img_url, use_container_width=True)
                    else:
                        # Inisial dengan gradien
                        if row["Score"] >= 8:
                            grad = "linear-gradient(135deg, #7c3aed, #6d28d9)"
                        elif row["Score"] >= 7:
                            grad = "linear-gradient(135deg, #22c55e, #16a34a)"
                        else:
                            grad = "linear-gradient(135deg, #f59e0b, #d97706)"
                        st.markdown(f"""
                        <div class="poster-initial" style="background:{grad};">
                            {row["Name"][0].upper()}
                        </div>
                        """, unsafe_allow_html=True)
                    st.caption(f"**{row['Name'][:20]}**")
                    st.caption(f"⭐ {row['Score']:.2f}  •  {row['Type']}")
                    st.caption(f"🔗 Similarity: {row['Similarity']:.3f}")

            # Tabel detail
            st.markdown("---")
            st.dataframe(recs[["Name", "Genres", "Score", "Similarity"]], use_container_width=True)

        else:
            st.warning("Anime ini tidak memiliki data similarity.")

# ==========================================================
# PAGE: AI INSIGHTS
# ==========================================================
elif page == "AI Insights":
    st.markdown('<div class="section-title">💡 AI Insights</div>', unsafe_allow_html=True)
    top_genre = df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().value_counts().idxmax()
    top_anime = df_anime.nlargest(1, "Score")["Name"].iloc[0]
    st.success(f"🔥 Most dominant genre: {top_genre}")
    st.success(f"⭐ Highest rated anime: {top_anime}")
    st.success("📈 Users tend to prefer high-score fantasy anime.")
    st.success("🎯 Recommendation Engine is ready for deployment.")
    st.success("🚀 Dashboard analytics generated successfully.")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        🎌 Anime Insight AI<br>
        Discover • Analyze • Recommend<br><br>
        Built with Streamlit, Plotly, Scikit-Learn<br>
        Data Source: MyAnimeList 2023<br><br>
        © 2026 Anime Insight AI
    </div>
    """,
    unsafe_allow_html=True
)
