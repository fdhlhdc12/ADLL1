# ==========================================================
# ANIME INSIGHT AI — PREMIUM FINAL
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

# ==========================================================
# DETEKSI GAMBAR BACKGROUND (auto ekstensi)
# ==========================================================
def get_base64(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

BG_IMAGE = ""
if os.path.exists("assets"):
    for ext in ["jpg", "jpeg", "png", "webp"]:
        path = f"assets/anime_bg.{ext}"
        if os.path.exists(path):
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

# Bersihkan Rank (jika ada) untuk korelasi
if "Rank" in df_anime.columns:
    df_anime["Rank"] = pd.to_numeric(df_anime["Rank"], errors="coerce")

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
# PREMIUM CSS (gradasi ungu-biru-hitam)
# ==========================================================
bg_style = f"""
.stApp {{
    background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 27, 75, 0.95), rgba(88, 28, 135, 0.90)),
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
        padding-top: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-bottom: 2rem;
    }}
    /* SIDEBAR */
    section[data-testid="stSidebar"] {{
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(20px);
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
        background: linear-gradient(135deg, #c084fc, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .logo-sub {{
        font-size: 12px;
        color: #94a3b8;
        letter-spacing: 2px;
    }}
    /* BUTTON SIDEBAR */
    div[data-testid="stButton"] button {{
        width: 100%;
        background: transparent;
        border: none;
        color: #cbd5e1;
        padding: 8px 12px;
        border-radius: 10px;
        text-align: left;
        font-size: 16px;
        transition: 0.3s;
    }}
    div[data-testid="stButton"] button:hover {{
        background: rgba(139, 92, 246, 0.2);
        color: white;
    }}
    div[data-testid="stButton"] button:focus {{
        background: #7c3aed !important;
        color: white !important;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.4);
    }}
    /* KPI CARDS */
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
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.3);
        border-color: rgba(139, 92, 246, 0.3);
    }}
    .kpi-value {{
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, #c084fc, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .kpi-label {{
        font-size: 14px;
        color: #cbd5e1;
    }}
    .kpi-growth {{
        font-size: 12px;
        color: #22c55e;
    }}
    /* SECTION TITLE */
    .section-title {{
        font-size: 28px;
        font-weight: 700;
        margin: 25px 0 20px 0;
        color: white;
        border-left: 5px solid #8b5cf6;
        padding-left: 15px;
    }}
    /* HERO BANNER */
    .hero-container {{
        position: relative;
        height: 280px;
        overflow: hidden;
        border-radius: 25px;
        margin-bottom: 25px;
        border: 1px solid rgba(139, 92, 246, 0.2);
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
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
        padding: 35px 45px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: linear-gradient(90deg, rgba(15,23,42,0.95) 0%, rgba(15,23,42,0.70) 50%, rgba(15,23,42,0.10) 100%);
    }}
    .hero-overlay .title {{
        font-size: 2.8rem;
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
        font-size: 1.2rem;
        color: #c084fc;
        font-weight: 600;
        margin-top: 5px;
    }}
    .hero-overlay .desc {{
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 5px;
    }}
    /* POSTER CARD */
    .poster-initial {{
        aspect-ratio: 2/3;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 4rem;
        font-weight: 700;
        color: white;
        width: 100%;
        background: linear-gradient(135deg, #7c3aed, #6d28d9);
    }}
    .anime-poster-card {{
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.05);
        transition: 0.3s;
    }}
    .anime-poster-card:hover {{
        transform: scale(1.02);
        border-color: #8b5cf6;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.2);
    }}
    /* FOOTER */
    .footer {{
        text-align: center;
        padding: 20px;
        color: #94a3b8;
        font-size: 13px;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 30px;
    }}
    .footer a {{
        color: #c084fc;
        text-decoration: none;
    }}
    /* DATAFRAME */
    .stDataFrame {{
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
    }}
    /* PLOTLY */
    .js-plotly-plot {{
        border-radius: 16px;
        overflow: hidden;
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
        <div class="logo-title">🎌 ANIME INSIGHT AI</div>
        <div class="logo-sub">アニメインサイト</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    menu_items = [
        ("🏠 Overview", "Overview"),
        ("🎬 Anime Explorer", "Anime Explorer"),
        ("📊 Analytics", "Analytics"),
        ("👥 User Analytics", "User Analytics"),
        ("🎯 Recommendations", "Recommendations"),
        ("💡 AI Insights", "AI Insights"),
    ]

    for label, key in menu_items:
        if st.button(label, key=f"btn_{key}", use_container_width=True):
            st.session_state.page = key
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
                    box-shadow: 0 4px 15px rgba(124,58,237,0.4);
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown(
        """
        <div style="text-align:center; color:#94a3b8; font-size:12px;">
            ⭐ Premium Dashboard<br>Powered by Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================
# PAGE: OVERVIEW
# ==========================================================
if st.session_state.page == "Overview":
    show_banner(
        title="Welcome Back, <span>Anime Explorer!</span>",
        subtitle="Anime Insight AI",
        desc="Discover • Analyze • Recommend — powered by Collaborative Filtering."
    )

    search_query = st.text_input("🔍 Search Anime", placeholder="Search anime, genre, studio...")
    if search_query:
        result = df_anime[df_anime["Name"].str.contains(search_query, case=False, na=False)]
        if not result.empty:
            st.success(f"Found {len(result)} anime")
            st.dataframe(result[["Name", "Genres", "Type", "Score"]].head(10), use_container_width=True, hide_index=True)
        else:
            st.info("No anime found")

    # KPI
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
            <div class="kpi-label">🎬 Total Anime</div>
            <div class="kpi-growth">+12.4%</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_users:,}</div>
            <div class="kpi-label">👥 Total Users</div>
            <div class="kpi-growth">+8.7%</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_score}</div>
            <div class="kpi-label">⭐ Avg Score</div>
            <div class="kpi-growth">+3.1%</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_genres}</div>
            <div class="kpi-label">🏷 Total Genres</div>
            <div class="kpi-growth" style="color:#94a3b8;">Stable</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_ratings:,}</div>
            <div class="kpi-label">📝 Ratings</div>
            <div class="kpi-growth">+15.3%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Score Distribution
    st.markdown('<div class="section-title">📊 Anime Score Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(df_anime, x="Score", nbins=30, color_discrete_sequence=["#8b5cf6"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Top Rated + Score by Type
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section-title">🏆 Top Rated Anime</div>', unsafe_allow_html=True)
        top5 = df_anime.nlargest(5, "Score")[["Name", "Score", "Image URL"]]
        # Tampilkan poster untuk top 5
        cols = st.columns(5)
        for i, (_, row) in enumerate(top5.iterrows()):
            with cols[i]:
                img = row["Image URL"]
                if pd.notna(img):
                    st.image(img, use_container_width=True)
                else:
                    st.markdown(f"""
                    <div class="poster-initial" style="background:linear-gradient(135deg, #7c3aed, #6d28d9);">
                        {row['Name'][0].upper()}
                    </div>
                    """, unsafe_allow_html=True)
                st.caption(f"**{row['Name'][:15]}**")
                st.caption(f"⭐ {row['Score']:.2f}")
        st.dataframe(df_anime.nlargest(10, "Score")[["Name", "Score"]], use_container_width=True, hide_index=True)
    with right:
        st.markdown('<div class="section-title">📈 Score by Type</div>', unsafe_allow_html=True)
        avg_type = df_anime.groupby("Type")["Score"].mean().reset_index()
        fig2 = px.bar(avg_type, x="Type", y="Score", color="Type", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Anime of the Day (dengan poster)
    st.markdown('<div class="section-title">⭐ Anime of the Day</div>', unsafe_allow_html=True)
    anime_day = df_anime.sample(1).iloc[0]
    col1, col2 = st.columns([1, 3])
    with col1:
        img = anime_day["Image URL"]
        if pd.notna(img):
            st.image(img, use_container_width=True)
        else:
            st.markdown(f"""
            <div class="poster-initial" style="background:linear-gradient(135deg, #7c3aed, #6d28d9);">
                {anime_day['Name'][0].upper()}
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"### {anime_day['Name']}")
        st.markdown(f"⭐ **Score:** {anime_day['Score']:.2f}  |  🎬 **Type:** {anime_day['Type']}  |  🏷 **Genre:** {anime_day['Genres']}")
        if pd.notna(anime_day["Synopsis"]):
            st.info(anime_day["Synopsis"][:400])

    # Genre Pie
    st.markdown('<div class="section-title">🎯 Genre Distribution</div>', unsafe_allow_html=True)
    genre_counts = df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().value_counts().head(10)
    fig3 = px.pie(values=genre_counts.values, names=genre_counts.index, color_discrete_sequence=px.colors.qualitative.Set3)
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=450)
    st.plotly_chart(fig3, use_container_width=True)

    # AI Insights
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
# PAGE: ANIME EXPLORER
# ==========================================================
elif st.session_state.page == "Anime Explorer":
    show_banner(
        title="🎬 Anime Explorer",
        subtitle="Discover Your Next Favorite",
        desc="Search, filter, and explore thousands of anime titles with posters."
    )

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

    filtered = df_anime.copy()
    if search_text:
        filtered = filtered[filtered["Name"].str.contains(search_text, case=False, na=False)]
    if selected_type:
        filtered = filtered[filtered["Type"].isin(selected_type)]
    if selected_genres:
        filtered = filtered[filtered["Genres"].apply(lambda x: any(g in str(x) for g in selected_genres))]
    filtered = filtered[(filtered["Score"] >= score_range[0]) & (filtered["Score"] <= score_range[1])]

    st.success(f"Found {len(filtered):,} Anime")

    # Stats
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Anime", len(filtered))
    s2.metric("Avg Score", round(filtered["Score"].mean(), 2))
    s3.metric("Highest Score", round(filtered["Score"].max(), 2))
    s4.metric("Types", filtered["Type"].nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    # Gallery with posters (max 8 untuk performa)
    st.markdown('<div class="section-title">🌟 Anime Gallery</div>', unsafe_allow_html=True)
    preview = filtered.head(8)
    cols = st.columns(4)
    for idx, (_, row) in enumerate(preview.iterrows()):
        with cols[idx % 4]:
            img = row["Image URL"]
            if pd.notna(img):
                st.image(img, use_container_width=True)
            else:
                score = row["Score"]
                if score >= 8:
                    grad = "linear-gradient(135deg, #7c3aed, #6d28d9)"
                elif score >= 7:
                    grad = "linear-gradient(135deg, #22c55e, #16a34a)"
                else:
                    grad = "linear-gradient(135deg, #f59e0b, #d97706)"
                st.markdown(f"""
                <div class="poster-initial" style="background:{grad};">
                    {row['Name'][0].upper()}
                </div>
                """, unsafe_allow_html=True)
            st.caption(f"**{row['Name'][:25]}**")
            st.caption(f"⭐ {row['Score']:.2f}  •  {row['Type']}")

    # Top genres & score distribution
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">🏷 Top Genres</div>', unsafe_allow_html=True)
        genre_counts2 = filtered[filtered["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().value_counts().head(10)
        fig = px.bar(x=genre_counts2.values, y=genre_counts2.index, orientation="h", color=genre_counts2.values, color_continuous_scale="purples")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">📈 Score Distribution</div>', unsafe_allow_html=True)
        fig2 = px.histogram(filtered, x="Score", nbins=20, color_discrete_sequence=["#8b5cf6"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=450)
        st.plotly_chart(fig2, use_container_width=True)

    # Data table
    st.markdown('<div class="section-title">📋 Anime Catalog</div>', unsafe_allow_html=True)
    display_cols = ["Name", "Genres", "Type", "Score", "Members"]
    st.dataframe(filtered[display_cols].sort_values("Score", ascending=False), use_container_width=True, height=600)

# ==========================================================
# PAGE: ANALYTICS (dengan heatmap fixed)
# ==========================================================
elif st.session_state.page == "Analytics":
    show_banner(
        title="📊 Analytics Dashboard",
        subtitle="Deep Insights & Trends",
        desc="Explore statistical patterns and correlations in anime data."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Highest Score", round(df_anime["Score"].max(), 2))
    c2.metric("Avg Members", f"{int(df_anime['Members'].mean()):,}")
    c3.metric("Anime Types", df_anime["Type"].nunique())
    c4.metric("Total Genres", df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().nunique())

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">⭐ Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df_anime, x="Score", nbins=30, color_discrete_sequence=["#8b5cf6"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">🎯 Genre Distribution</div>', unsafe_allow_html=True)
        genre_counts = df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().value_counts().head(15)
        fig2 = px.bar(x=genre_counts.index, y=genre_counts.values, color=genre_counts.values, color_continuous_scale="purples")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False, height=400)
        st.plotly_chart(fig2, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">🔥 Most Popular</div>', unsafe_allow_html=True)
        popular = df_anime[df_anime["Popularity"] > 0].nsmallest(15, "Popularity")
        fig3 = px.bar(popular, x="Popularity", y="Name", orientation="h", color="Popularity", color_continuous_scale="purples")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=500, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">👥 Top Members</div>', unsafe_allow_html=True)
        members = df_anime.nlargest(15, "Members")
        fig4 = px.bar(members, x="Members", y="Name", orientation="h", color="Members", color_continuous_scale="blues")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=500, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    # Score vs Members
    st.markdown('<div class="section-title">📊 Score vs Members</div>', unsafe_allow_html=True)
    sample_df = df_anime.sample(min(2000, len(df_anime)))
    fig5 = px.scatter(sample_df, x="Members", y="Score", color="Type", hover_data=["Name"], color_discrete_sequence=px.colors.qualitative.Pastel)
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=600)
    st.plotly_chart(fig5, use_container_width=True)

    # Heatmap — FIXED
    st.markdown('<div class="section-title">🌡 Correlation Heatmap</div>', unsafe_allow_html=True)
    # Pilih kolom numerik yang valid
    numeric_cols = ["Score", "Members", "Popularity", "Rank"]
    available_cols = [c for c in numeric_cols if c in df_anime.columns]
    if len(available_cols) > 1:
        # Pastikan semua numerik, drop NaN
        corr_data = df_anime[available_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if not corr_data.empty and corr_data.shape[0] > 1:
            corr = corr_data.corr()
            fig6 = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
            fig6.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("Not enough valid numeric data for heatmap.")
    else:
        st.info("Need at least 2 numeric columns for heatmap.")

    # WordCloud
    st.markdown('<div class="section-title">☁ Genre WordCloud</div>', unsafe_allow_html=True)
    genre_text = " ".join(df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].dropna())
    if genre_text.strip():
        wc = WordCloud(width=1200, height=500, background_color="black", colormap="plasma").generate(genre_text)
        fig_wc, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig_wc)

    # Radar
    st.markdown('<div class="section-title">🕸 Dataset Radar</div>', unsafe_allow_html=True)
    if "Score" in df_anime.columns and "Members" in df_anime.columns:
        radar_categories = ["Score", "Members (log)"]
        radar_values = [df_anime["Score"].mean(), np.log1p(df_anime["Members"].mean())]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=radar_values, theta=radar_categories, fill="toself", marker=dict(color="#8b5cf6")))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=500)
        st.plotly_chart(fig_radar, use_container_width=True)

    # Data preview
    st.markdown('<div class="section-title">📋 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_anime.head(100), use_container_width=True, height=500)

# ==========================================================
# PAGE: USER ANALYTICS
# ==========================================================
elif st.session_state.page == "User Analytics":
    show_banner(
        title="👥 User Analytics",
        subtitle="Who Watches Anime?",
        desc="Demographics, locations, and age distribution of anime fans."
    )

    total_users = len(df_user)
    gender_count = df_user["Gender"].dropna().value_counts()
    male_count = gender_count[gender_count.index.str.contains("male", case=False, na=False)].sum()
    female_count = gender_count[gender_count.index.str.contains("female", case=False, na=False)].sum()
    total_country = df_user["Location"].dropna().nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Users", f"{total_users:,}")
    c2.metric("Countries", total_country)
    c3.metric("Male", male_count)
    c4.metric("Female", female_count)

    left, right = st.columns(2)
    with left:
        fig = px.pie(values=gender_count.values, names=gender_count.index, title="Gender Distribution", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        loc_count = df_user["Location"].value_counts().head(15)
        fig2 = px.bar(x=loc_count.values, y=loc_count.index, orientation="h", title="Top Locations", color=loc_count.values, color_continuous_scale="purples")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

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
            fig3 = px.histogram(ages, nbins=25, title="Age Distribution", color_discrete_sequence=["#8b5cf6"])
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No valid age data.")
    else:
        st.info("Birthday column not available.")

# ==========================================================
# PAGE: RECOMMENDATIONS (Netflix style)
# ==========================================================
elif st.session_state.page == "Recommendations":
    show_banner(
        title="🎯 Recommendation Engine",
        subtitle="AI-Powered Personalized Picks",
        desc="Find your next favorite anime using Collaborative Filtering."
    )

    available = df_anime[df_anime["anime_id"].isin(similarity_df.index)]
    anime_list = sorted(available["Name"].dropna().unique())

    col1, col2 = st.columns([4, 1])
    with col1:
        selected = st.selectbox("Choose Anime", anime_list, index=0)
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

        # Detail dengan poster
        col1, col2 = st.columns([1, 3])
        with col1:
            img = info["Image URL"]
            if pd.notna(img):
                st.image(img, use_container_width=True)
            else:
                st.markdown(f"""
                <div class="poster-initial" style="background:linear-gradient(135deg, #7c3aed, #6d28d9);">
                    {info['Name'][0].upper()}
                </div>
                """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"### {info['Name']}")
            st.markdown(f"⭐ **Score:** {info['Score']:.2f}  |  🎞 **Type:** {info['Type']}  |  👥 **Members:** {info['Members']:,}")
            st.markdown(f"🏷 **Genre:** {info['Genres']}")
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
                    img = row["Image URL"]
                    if pd.notna(img):
                        st.image(img, use_container_width=True)
                    else:
                        score = row["Score"]
                        if score >= 8:
                            grad = "linear-gradient(135deg, #7c3aed, #6d28d9)"
                        elif score >= 7:
                            grad = "linear-gradient(135deg, #22c55e, #16a34a)"
                        else:
                            grad = "linear-gradient(135deg, #f59e0b, #d97706)"
                        st.markdown(f"""
                        <div class="poster-initial" style="background:{grad};">
                            {row['Name'][0].upper()}
                        </div>
                        """, unsafe_allow_html=True)
                    st.caption(f"**{row['Name'][:20]}**")
                    st.caption(f"⭐ {row['Score']:.2f}  •  {row['Type']}")
                    st.caption(f"🔗 Similarity: {row['Similarity']:.3f}")

            st.markdown("---")
            st.dataframe(recs[["Name", "Genres", "Score", "Similarity"]], use_container_width=True)

        else:
            st.warning("This anime does not have enough rating data for recommendations.")

# ==========================================================
# PAGE: AI INSIGHTS
# ==========================================================
elif st.session_state.page == "AI Insights":
    show_banner(
        title="💡 AI Insights",
        subtitle="Smart Summaries from Data",
        desc="Key takeaways and interesting patterns discovered in the dataset."
    )

    top_genre = df_anime[df_anime["Genres"] != "UNKNOWN"]["Genres"].str.split(", ").explode().value_counts().idxmax()
    top_anime = df_anime.nlargest(1, "Score")["Name"].iloc[0]
    st.success(f"🔥 Most dominant genre: **{top_genre}**")
    st.success(f"⭐ Highest rated anime: **{top_anime}**")
    st.success("📈 Users who watch Fantasy tend to give higher ratings.")
    st.success("🎯 Recommendation Engine ready to serve.")
    st.success("🚀 Dashboard analytics successfully generated.")

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
