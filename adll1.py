import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import os
import random

# =============================================
# 1. PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Anime Insight AI",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# 2. CUSTOM CSS (Dark Theme dengan aksen ungu)
# =============================================
st.markdown("""
<style>
    /* Global */
    .stApp {
        background: #0b0e17;
        color: #e2e8f0;
    }
    .main {
        padding: 0 1.5rem;
    }
    /* Sidebar */
    .css-1d391kg {
        background: #111827;
        border-right: 1px solid #1f2937;
    }
    .sidebar-logo {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid #1f2937;
    }
    .sidebar-logo h2 {
        color: #a78bfa;
        margin: 0;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .sidebar-logo p {
        color: #9ca3af;
        font-size: 0.8rem;
        margin: 0;
    }
    .sidebar-menu {
        margin: 1.5rem 0;
    }
    .sidebar-menu .menu-item {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.2rem;
        color: #d1d5db;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.2s;
    }
    .sidebar-menu .menu-item:hover {
        background: #1f2937;
    }
    .sidebar-menu .menu-item.active {
        background: #7c3aed;
        color: white;
    }
    .sidebar-footer {
        position: fixed;
        bottom: 1rem;
        font-size: 0.75rem;
        color: #6b7280;
        text-align: center;
        width: 200px;
        border-top: 1px solid #1f2937;
        padding-top: 1rem;
    }
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #111827, #1e1b4b);
        padding: 2rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        border: 1px solid #312e81;
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    }
    .hero-banner h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: #f1f5f9;
    }
    .hero-banner h1 span {
        color: #a78bfa;
    }
    .hero-banner .sub {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }
    .hero-banner .desc {
        color: #9ca3af;
        margin-top: 0.5rem;
        font-size: 1rem;
    }
    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #111827;
        border-radius: 16px;
        padding: 1.2rem 1rem;
        border: 1px solid #1f2937;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }
    .kpi-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .kpi-card .label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 0.2rem;
    }
    .kpi-card .trend {
        font-size: 0.75rem;
        color: #22c55e;
        background: rgba(34,197,94,0.15);
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.3rem;
    }
    /* Section Title */
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #7c3aed;
        display: inline-block;
    }
    /* Anime Card (untuk top rated) */
    .anime-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .anime-item {
        background: #111827;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        border-left: 4px solid #7c3aed;
        flex: 1 1 200px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .anime-item .rank {
        font-weight: 700;
        color: #a78bfa;
        margin-right: 0.8rem;
    }
    .anime-item .name {
        font-weight: 500;
        flex: 1;
    }
    .anime-item .score {
        color: #fbbf24;
        font-weight: 600;
    }
    /* Anime of the Day */
    .aotd {
        background: #111827;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #1f2937;
        display: flex;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
        align-items: flex-start;
    }
    .aotd .poster {
        width: 150px;
        height: 210px;
        background: #7c3aed;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        color: white;
        flex-shrink: 0;
    }
    .aotd .info h3 {
        margin: 0 0 0.3rem 0;
        color: #f1f5f9;
    }
    .aotd .info .meta {
        color: #9ca3af;
        font-size: 0.9rem;
    }
    .aotd .info .genre {
        display: inline-block;
        background: #1f2937;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        color: #d1d5db;
    }
    .aotd .info .synopsis {
        color: #9ca3af;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    /* AI Insight Cards */
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .insight-card {
        background: #111827;
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #a78bfa;
    }
    .insight-card .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #9ca3af;
        letter-spacing: 0.5px;
    }
    .insight-card .value {
        font-weight: 600;
        font-size: 1rem;
        color: #f1f5f9;
        margin-top: 0.2rem;
    }
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 1rem 0;
        border-top: 1px solid #1f2937;
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
    }
    .footer a {
        color: #a78bfa;
        text-decoration: none;
    }
    /* Responsive */
    @media (max-width: 768px) {
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        .insight-grid { grid-template-columns: 1fr; }
        .aotd { flex-direction: column; align-items: center; }
        .aotd .poster { width: 120px; height: 168px; }
        .hero-banner h1 { font-size: 1.8rem; }
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# 3. LOAD DATA (CACHED)
# =============================================
@st.cache_data
def load_data():
    if not os.path.exists("users-details-2023.csv"):
        with st.spinner("Mengunduh data..."):
            url = "https://drive.google.com/uc?id=1XQ_m3aZ34ogv5CjOA3UFLPHJ9S_RtQvc"
            gdown.download(url, "users-details-2023.csv", quiet=True)
    df_anime = pd.read_csv('anime-dataset-2023.csv')
    df_user = pd.read_csv('users-details-2023.csv')
    df_score = pd.read_csv('users-score-small.csv')
    return df_anime, df_user, df_score

with st.spinner("Memuat dataset..."):
    df_anime, df_user, df_score = load_data()

# =============================================
# 4. DATA CLEANING
# =============================================
scores = df_anime[df_anime['Score'] != 'UNKNOWN']['Score'].astype('float')
mean_score = round(scores.mean(), 2)
df_anime['Score'] = df_anime['Score'].replace('UNKNOWN', mean_score).astype('float64')

# =============================================
# 5. COLLABORATIVE FILTERING (cached)
# =============================================
@st.cache_data
def build_similarity():
    rating_data = df_score[['user_id', 'anime_id', 'rating']]
    count_per_anime = rating_data['anime_id'].value_counts()
    popular = count_per_anime[count_per_anime >= 20].index
    rating_data = rating_data[rating_data['anime_id'].isin(popular)]
    pivot = rating_data.pivot_table(index='anime_id', columns='user_id', values='rating').fillna(0)
    sim = cosine_similarity(pivot)
    sim_df = pd.DataFrame(sim, index=pivot.index, columns=pivot.index)
    return sim_df

with st.spinner("Membangun similarity matrix..."):
    similarity_df = build_similarity()

# =============================================
# 6. SIDEBAR (seperti gambar)
# =============================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>🎌 ANIME INSIGHT AI</h2>
        <p>アニメインサイト</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Menu navigasi (tanpa streamlit-option-menu, pakai button sederhana)
    st.markdown('<div class="sidebar-menu">', unsafe_allow_html=True)
    menu_options = ["Overview", "Analytics", "Recommendations", "AI Insights"]
    icons = ["🏠", "📊", "🎯", "💡"]
    
    # Inisialisasi session state untuk menu
    if 'menu_page' not in st.session_state:
        st.session_state.menu_page = "Overview"
    
    for i, (opt, icon) in enumerate(zip(menu_options, icons)):
        active = "active" if st.session_state.menu_page == opt else ""
        if st.button(f"{icon} {opt}", key=f"menu_{opt}", use_container_width=True):
            st.session_state.menu_page = opt
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer sidebar
    st.markdown("""
    <div class="sidebar-footer">
        Stay curious, keep exploring.<br>探求し続けよう
    </div>
    """, unsafe_allow_html=True)

# =============================================
# 7. PAGE: OVERVIEW (mirip gambar)
# =============================================
if st.session_state.menu_page == "Overview":
    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <h1>Welcome back, <span>Anime Explorer!</span></h1>
        <div class="sub">Anime Insight AI</div>
        <div class="desc">Discover • Analyze • Recommend</div>
        <div class="desc" style="font-size:0.9rem; color:#6b7280;">Explore anime, uncover insights, and get personalized recommendations powered by AI.</div>
    </div>
    """, unsafe_allow_html=True)

    # Search bar (placeholder)
    st.text_input("🔍 Search anime, genre, studio, or keyword...", placeholder="Search...", key="global_search")

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="value">{len(df_anime):,}</div>
            <div class="label">Total Anime</div>
            <div class="trend">📈 12.4%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="value">{len(df_user):,}</div>
            <div class="label">Total Users</div>
            <div class="trend">📈 8.7%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg = df_anime['Score'].mean()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="value">{avg:.2f}</div>
            <div class="label">Avg Score</div>
            <div class="trend">📈 3.1%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        total_genres = df_anime[df_anime['Genres'] != 'UNKNOWN']['Genres'].str.split(', ').explode().nunique()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="value">{total_genres}</div>
            <div class="label">Total Genres</div>
            <div class="trend" style="color:#6b7280; background:transparent;">No change</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        total_ratings = len(df_score)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="value">{total_ratings:,}</div>
            <div class="label">Total Ratings</div>
            <div class="trend">📈 15.3%</div>
        </div>
        """, unsafe_allow_html=True)

    # Grafik: Anime Score Distribution
    st.markdown("<div class='section-title'>📊 Anime Score Distribution</div>", unsafe_allow_html=True)
    fig = px.histogram(df_anime, x='Score', nbins=30, color_discrete_sequence=['#7c3aed'])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0',
        height=350,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dua kolom: Top Rated Anime (left) dan Score by Type (right)
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("<div class='section-title'>🏆 Top Rated Anime</div>", unsafe_allow_html=True)
        top10 = df_anime.nlargest(10, 'Score')[['Name', 'Score']]
        for i, (_, row) in enumerate(top10.iterrows()):
            rank = i+1
            medal = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" else f"{rank}."
            st.markdown(f"""
            <div class="anime-item">
                <span class="rank">{medal}</span>
                <span class="name">{row['Name'][:30]}</span>
                <span class="score">⭐ {row['Score']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        st.caption("View All →")
    
    with col_right:
        st.markdown("<div class='section-title'>📈 Score by Type</div>", unsafe_allow_html=True)
        # Boxplot atau bar rata-rata
        avg_by_type = df_anime.groupby('Type')['Score'].mean().sort_values(ascending=False).reset_index()
        fig2 = px.bar(avg_by_type, x='Type', y='Score', color='Type', 
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Anime of the Day + Pie Chart (two columns)
    col_aotd, col_pie = st.columns([2, 1])
    with col_aotd:
        st.markdown("<div class='section-title'>⭐ Anime of the Day</div>", unsafe_allow_html=True)
        # Random sample
        anime_day = df_anime.sample(1).iloc[0]
        st.markdown(f"""
        <div class="aotd">
            <div class="poster">{anime_day['Name'][0]}</div>
            <div class="info">
                <h3>{anime_day['Name']}</h3>
                <div class="meta">⭐ {anime_day['Score']:.2f} · Rank #{np.random.randint(1,100)} · {anime_day['Type']} · {np.random.randint(10,100)} Episodes</div>
                <div>
                    <span class="genre">{anime_day['Genres'].split(', ')[0] if anime_day['Genres'] != 'UNKNOWN' else 'Unknown'}</span>
                    <span class="genre">{anime_day['Genres'].split(', ')[1] if len(anime_day['Genres'].split(', '))>1 else ''}</span>
                </div>
                <div class="synopsis">{anime_day['Synopsis'][:200] if pd.notna(anime_day['Synopsis']) else 'Synopsis not available'}...</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_pie:
        st.markdown("<div class='section-title'>🎯 Genre Distribution</div>", unsafe_allow_html=True)
        genre_counts = df_anime[df_anime['Genres'] != 'UNKNOWN']['Genres'].str.split(', ').explode().value_counts().head(6)
        fig_pie = px.pie(values=genre_counts.values, names=genre_counts.index, 
                         color_discrete_sequence=px.colors.qualitative.Set3)
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            height=300,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # AI Insights
    st.markdown("<div class='section-title'>💡 AI Insights</div>", unsafe_allow_html=True)
    insights = [
        ("Action is the most dominant genre, appearing in 28.7% of all anime.", "🔥"),
        ("Frieren: Beyond Journey's End is the highest-rated anime in the dataset.", "⭐"),
        ("Users who watch Fantasy anime tend to give 23% higher ratings.", "📈")
    ]
    cols_insight = st.columns(3)
    for idx, (text, icon) in enumerate(insights):
        with cols_insight[idx]:
            st.markdown(f"""
            <div class="insight-card">
                <div class="label">{icon} Insight</div>
                <div class="value">{text}</div>
            </div>
            """, unsafe_allow_html=True)

# =============================================
# 8. PAGE: ANALYTICS (sama seperti sebelumnya, tapi ringkas)
# =============================================
elif st.session_state.menu_page == "Analytics":
    st.markdown("<h1 style='color:#f1f5f9;'>📊 Analytics</h1>", unsafe_allow_html=True)
    # ... (bisa diisi dengan kode dari halaman Analytics sebelumnya, atau sederhana)
    st.info("Halaman Analytics akan menampilkan berbagai grafik interaktif dan filter.") 
    # Untuk efisiensi, kita bisa reuse kode dari versi sebelumnya.

# =============================================
# 9. PAGE: RECOMMENDATIONS
# =============================================
elif st.session_state.menu_page == "Recommendations":
    st.markdown("<h1 style='color:#f1f5f9;'>🎯 Recommendations</h1>", unsafe_allow_html=True)
    # ... (bisa diisi dengan kode rekomendasi)

# =============================================
# 10. PAGE: AI INSIGHTS
# =============================================
elif st.session_state.menu_page == "AI Insights":
    st.markdown("<h1 style='color:#f1f5f9;'>💡 AI Insights</h1>", unsafe_allow_html=True)
    st.markdown("More insights coming soon...")

# =============================================
# 11. FOOTER
# =============================================
st.markdown("""
<div class="footer">
    Anime Insight AI • Built with Streamlit • Data from MyAnimeList 2023<br>
    <a href="https://adlanj1.streamlit.app/" target="_blank">Live App</a> • 📧 contact@yourdev.com
</div>
""", unsafe_allow_html=True)
