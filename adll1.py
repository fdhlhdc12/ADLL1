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

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Anime Insight Recommendation",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CUSTOM CSS (Premium)
# =====================================
st.markdown("""
<style>
    /* Global */
    .stApp {
        background: #f4f6f9;
    }
    .main {
        padding: 0 1rem;
    }
    /* Sidebar */
    .css-1d391kg {
        background: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    /* Cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.2rem 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s;
        border-top: 5px solid #ff6b6b;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.10);
    }
    .metric-card .number {
        font-size: 2.4rem;
        font-weight: 700;
        color: #2d3436;
    }
    .metric-card .label {
        font-size: 0.9rem;
        color: #636e72;
        margin-top: 0.2rem;
    }
    .metric-card .icon {
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    .section-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2d3436;
        margin: 1.8rem 0 1.2rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 4px solid #ff6b6b;
        display: inline-block;
    }
    .anime-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 5px solid #6c5ce7;
        margin-bottom: 0.8rem;
        transition: all 0.2s;
    }
    .anime-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        transform: translateX(4px);
    }
    .anime-card .title {
        font-weight: 600;
        font-size: 1rem;
        color: #2d3436;
    }
    .anime-card .detail {
        font-size: 0.85rem;
        color: #636e72;
    }
    .footer {
        margin-top: 3rem;
        padding: 1.5rem 0;
        border-top: 1px solid #e9ecef;
        text-align: center;
        color: #b2bec3;
        font-size: 0.9rem;
    }
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        font-weight: 500;
        color: #2d3436;
    }
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2d3436;
    }
</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATA (with caching & progress)
# =====================================
@st.cache_data
def download_and_load():
    # Download jika belum ada
    if not os.path.exists("users-details-2023.csv"):
        with st.spinner("Mengunduh data... (3 file)"):
            url = "https://drive.google.com/uc?id=1XQ_m3aZ34ogv5CjOA3UFLPHJ9S_RtQvc"
            gdown.download(url, "users-details-2023.csv", quiet=True)
    # Load
    df_anime = pd.read_csv('anime-dataset-2023.csv')
    df_user = pd.read_csv('users-details-2023.csv')
    df_score = pd.read_csv('users-score-small.csv')
    return df_anime, df_user, df_score

with st.spinner("Memuat dataset..."):
    df_anime, df_user, df_score = download_and_load()

# =====================================
# DATA CLEANING
# =====================================
# Score
scores = df_anime[df_anime['Score'] != 'UNKNOWN']['Score'].astype('float')
score_mean = round(scores.mean(), 2)
df_anime['Score'] = df_anime['Score'].replace('UNKNOWN', score_mean).astype('float64')

# =====================================
# COLLABORATIVE FILTERING (CACHED)
# =====================================
@st.cache_data
def build_similarity():
    rating_data = df_score[['user_id', 'anime_id', 'rating']]
    anime_rating_count = rating_data['anime_id'].value_counts()
    popular_anime = anime_rating_count[anime_rating_count >= 20].index
    rating_data = rating_data[rating_data['anime_id'].isin(popular_anime)]
    anime_pivot = rating_data.pivot_table(
        index='anime_id',
        columns='user_id',
        values='rating'
    ).fillna(0)
    anime_sim = cosine_similarity(anime_pivot)
    sim_df = pd.DataFrame(
        anime_sim,
        index=anime_pivot.index,
        columns=anime_pivot.index
    )
    return sim_df

with st.spinner("Membangun similarity matrix..."):
    similarity_df = build_similarity()

# =====================================
# SIDEBAR NAVIGATION
# =====================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/anime.png", width=80)
    st.title("🎌 Menu")
    st.markdown("---")
    menu = st.radio(
        "Navigasi",
        ["🏠 Overview", "📊 Anime Analysis", "👥 User Analysis", "🎯 Recommendation"],
        index=0
    )
    st.markdown("---")
    st.caption(f"📅 Data: {datetime.now().strftime('%d %b %Y')}")
    st.caption("⚙️ v2.0 - Premium")

# =====================================
# PAGE: OVERVIEW
# =====================================
if menu == "🏠 Overview":
    st.markdown("<h1 style='color:#2d3436;'>🎌 Anime Insight Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#636e72; font-size:1.1rem;'>Sistem rekomendasi cerdas berbasis <strong>Collaborative Filtering</strong> dengan visualisasi interaktif</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color:#ff6b6b;">
            <div class="icon">🎬</div>
            <div class="number">{len(df_anime):,}</div>
            <div class="label">Total Anime</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color:#4ecdc4;">
            <div class="icon">👤</div>
            <div class="number">{len(df_user):,}</div>
            <div class="label">Total Users</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color:#fdcb6e;">
            <div class="icon">⭐</div>
            <div class="number">{len(df_score):,}</div>
            <div class="label">Total Ratings</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        avg_score = df_anime['Score'].mean()
        st.markdown(f"""
        <div class="metric-card" style="border-top-color:#6c5ce7;">
            <div class="icon">📊</div>
            <div class="number">{avg_score:.2f}</div>
            <div class="label">Rata-rata Skor</div>
        </div>
        """, unsafe_allow_html=True)

    # Additional stats
    col5, col6, col7 = st.columns(3)
    with col5:
        total_genres = df_anime[df_anime['Genres'] != 'UNKNOWN']['Genres'].str.split(', ').explode().nunique()
        st.metric("Genre Unik", total_genres)
    with col6:
        total_types = df_anime['Type'].nunique()
        st.metric("Tipe Anime", total_types)
    with col7:
        avg_members = int(df_anime['Members'].mean())
        st.metric("Rata-rata Member", f"{avg_members:,}")

    st.markdown("---")

    # Quick preview of top anime
    st.markdown("<h3 class='section-title'>🏆 Top 6 Anime by Score</h3>", unsafe_allow_html=True)
    top6 = df_anime.nlargest(6, 'Score')[['Name', 'Genres', 'Score', 'Type']]
    cols = st.columns(3)
    for i, (_, row) in enumerate(top6.iterrows()):
        with cols[i % 3]:
            color = ['#ff6b6b','#4ecdc4','#45b7d1','#f9ca24','#6c5ce7','#fd79a8'][i]
            st.markdown(f"""
            <div style="background:{color}; border-radius:12px; padding:1.2rem; color:white; margin-bottom:0.8rem; text-align:center;">
                <div style="font-size:1.8rem;">🎬</div>
                <div style="font-weight:600; font-size:0.95rem;">{row['Name'][:30]}{'...' if len(row['Name'])>30 else ''}</div>
                <div style="font-size:0.75rem; opacity:0.9;">⭐ {row['Score']:.2f} · {row['Type']}</div>
                <div style="font-size:0.7rem; opacity:0.7;">{row['Genres'][:40]}</div>
            </div>
            """, unsafe_allow_html=True)

    # About section
    with st.expander("📖 Tentang Dashboard Ini", expanded=False):
        st.markdown("""
        **Filosofi Desain**: Dashboard ini dibangun untuk menjawab tantangan *Paradox of Choice* di industri anime, 
        menggunakan data nyata dari ribuan pengguna untuk memberikan saran yang relevan secara matematis.

        **Algoritma**: Collaborative Filtering dengan Cosine Similarity. 
        Sistem menganalisis pola rating dari 20+ pengguna per anime untuk menemukan kemiripan.

        **Data**: Bersumber dari MyAnimeList 2023, dikurasi secara ketat.
        """)

# =====================================
# PAGE: ANIME ANALYSIS
# =====================================
elif menu == "📊 Anime Analysis":
    st.markdown("<h1 style='color:#2d3436;'>📊 Anime Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#636e72;'>Eksplorasi data anime dengan filter interaktif</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Filters
    with st.expander("🔍 Filter Data", expanded=True):
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            genres_all = df_anime[df_anime['Genres'] != 'UNKNOWN']['Genres'].str.split(', ').explode().unique()
            selected_genres = st.multiselect("Genre", sorted(genres_all), default=[])
        with col_f2:
            types_all = df_anime['Type'].unique()
            selected_types = st.multiselect("Tipe", sorted(types_all), default=[])
        with col_f3:
            score_range = st.slider("Rentang Skor", 0.0, 10.0, (0.0, 10.0), 0.1)
        with col_f4:
            min_members = st.number_input("Min Members", min_value=0, value=0, step=1000)

    # Filter data
    df_filtered = df_anime.copy()
    if selected_genres:
        df_filtered = df_filtered[df_filtered['Genres'].apply(
            lambda x: any(g in x for g in selected_genres) if x != 'UNKNOWN' else False
        )]
    if selected_types:
        df_filtered = df_filtered[df_filtered['Type'].isin(selected_types)]
    df_filtered = df_filtered[(df_filtered['Score'] >= score_range[0]) & (df_filtered['Score'] <= score_range[1])]
    if min_members > 0:
        df_filtered = df_filtered[df_filtered['Members'] >= min_members]

    st.info(f"Menampilkan {len(df_filtered)} anime dari {len(df_anime)} total")

    # Visualizations
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        # Score distribution
        fig = px.histogram(df_filtered, x='Score', nbins=30, title='Distribusi Skor', color_discrete_sequence=['#6c5ce7'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col_v2:
        # Score vs Members
        sample = df_filtered.sample(min(500, len(df_filtered)))
        fig = px.scatter(sample, x='Score', y='Members', color='Type', title='Score vs Members', hover_data=['Name'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    col_v3, col_v4 = st.columns(2)
    with col_v3:
        # Top popular
        top_pop = df_filtered[df_filtered['Popularity'] > 0].nsmallest(15, 'Popularity')
        fig = px.bar(top_pop, x='Name', y='Popularity', color='Name', title='Top 15 Popularity')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col_v4:
        # Top genres
        genre_counts = df_filtered[df_filtered['Genres'] != 'UNKNOWN']['Genres'].str.split(', ').explode().value_counts().head(20)
        fig = px.bar(genre_counts, x=genre_counts.index, y=genre_counts.values, color=genre_counts.index,
                     title='Top 20 Genres')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # WordCloud
    st.markdown("<h3 class='section-title'>☁️ Genre WordCloud</h3>", unsafe_allow_html=True)
    genre_text = ' '.join(df_filtered[df_filtered['Genres'] != 'UNKNOWN']['Genres'].dropna())
    if genre_text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(genre_text)
        fig_wc, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wordcloud)
        ax.axis('off')
        st.pyplot(fig_wc)
    else:
        st.warning("Tidak ada genre untuk ditampilkan setelah filter.")

    # Data table
    with st.expander("📋 Lihat Data Anime (Filtered)", expanded=False):
        st.dataframe(df_filtered[['Name', 'Genres', 'Type', 'Score', 'Members', 'Popularity']], use_container_width=True)

# =====================================
# PAGE: USER ANALYSIS
# =====================================
elif menu == "👥 User Analysis":
    st.markdown("<h1 style='color:#2d3436;'>👥 User Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#636e72;'>Demografi dan perilaku pengguna</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        genders = df_user['Gender'].dropna().unique()
        selected_gender = st.multiselect("Gender", sorted(genders), default=[])
    with col_f2:
        top_locs = df_user['Location'].value_counts().head(10).index
        selected_locs = st.multiselect("Lokasi (Top 10)", sorted(top_locs), default=[])

    df_user_filtered = df_user.copy()
    if selected_gender:
        df_user_filtered = df_user_filtered[df_user_filtered['Gender'].isin(selected_gender)]
    if selected_locs:
        df_user_filtered = df_user_filtered[df_user_filtered['Location'].isin(selected_locs)]

    st.info(f"Menampilkan {len(df_user_filtered)} user")

    col1, col2 = st.columns(2)
    with col1:
        # Gender pie
        gender_counts = df_user_filtered['Gender'].value_counts(dropna=True)
        if not gender_counts.empty:
            fig = px.pie(values=gender_counts.values, names=gender_counts.index, title='Gender Distribution',
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada data gender setelah filter")
    with col2:
        # Location bar
        loc_counts = df_user_filtered['Location'].value_counts().head(15)
        if not loc_counts.empty:
            fig = px.bar(loc_counts, x=loc_counts.index, y=loc_counts.values, color=loc_counts.index,
                         title='Top Locations')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Age distribution
    st.markdown("<h3 class='section-title'>📅 Age Distribution</h3>", unsafe_allow_html=True)
    def calc_age(birth):
        if birth != 'NaN':
            try:
                year = int(birth.split('-')[0])
                age = datetime.now().year - year
                if 10 <= age < 60:
                    return age
            except:
                pass
        return None
    ages = df_user_filtered['Birthday'].dropna().apply(calc_age)
    ages = ages.dropna()
    if not ages.empty:
        fig = px.histogram(ages, nbins=20, title='Age Distribution', color_discrete_sequence=['#ff6b6b'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ada data usia yang valid")

# =====================================
# PAGE: RECOMMENDATION
# =====================================
elif menu == "🎯 Recommendation":
    st.markdown("<h1 style='color:#2d3436;'>🎯 Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#636e72;'>Temukan anime serupa dengan Collaborative Filtering</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Explain
    with st.expander("💡 Bagaimana Sistem Bekerja", expanded=False):
        st.markdown("""
        Sistem menggunakan **Collaborative Filtering** berbasis **Cosine Similarity**.
        - Menganalisis pola rating dari **minimal 20 pengguna** per anime.
        - Membangun matriks kemiripan antar anime.
        - Ketika Anda memilih anime, sistem mencari 10 anime dengan skor kemiripan tertinggi.
        - **Skor kemiripan** menunjukkan seberapa besar kesamaan preferensi pengguna.
        """)

    # Select anime
    available_anime = df_anime[df_anime['anime_id'].isin(similarity_df.index)]
    anime_list = sorted(available_anime['Name'].dropna().unique())

    col_select, col_random = st.columns([3, 1])
    with col_select:
        selected = st.selectbox("🔍 Pilih Anime Favorit", ["-- Pilih --"] + anime_list, index=0)
    with col_random:
        if st.button("🎲 Random Anime"):
            selected = random.choice(anime_list)
            # Need to rerun to update selectbox, but streamlit doesn't allow dynamic default easily.
            # We'll use session state.
            if 'random_anime' not in st.session_state:
                st.session_state.random_anime = random.choice(anime_list)
            else:
                st.session_state.random_anime = random.choice(anime_list)
            # force rerun
            st.experimental_rerun()

    # If random selected, set the selectbox value via session state trick
    if 'random_anime' in st.session_state:
        selected = st.session_state.random_anime

    if selected != "-- Pilih --" and selected in anime_list:
        anime_info = df_anime[df_anime['Name'] == selected].iloc[0]
        anime_id = anime_info['anime_id']

        # Display detail
        st.markdown(f"""
        <div style="background:white; border-radius:12px; padding:1.2rem; box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-bottom:1.2rem;">
            <h3 style="color:#2d3436;">📺 {anime_info['Name']}</h3>
            <p><strong>Genre:</strong> {anime_info['Genres']}</p>
            <p><strong>Skor:</strong> ⭐ {anime_info['Score']:.2f} &nbsp;|&nbsp; <strong>Tipe:</strong> {anime_info['Type']}</p>
            <p><strong>Member:</strong> {anime_info['Members']:,}</p>
            <p><strong>Sinopsis:</strong><br>{anime_info['Synopsis'] if pd.notna(anime_info['Synopsis']) else 'Tidak tersedia'}</p>
        </div>
        """, unsafe_allow_html=True)

        # Get recommendations
        if anime_id in similarity_df.index:
            sim_scores = similarity_df[anime_id].sort_values(ascending=False)
            top_ids = sim_scores.iloc[1:11].index
            recs = df_anime[df_anime['anime_id'].isin(top_ids)][['Name', 'Genres', 'Score', 'Type']]
            # Add similarity score
            recs['Similarity'] = recs['anime_id'].map(lambda x: sim_scores[x] if x in sim_scores else 0)
            recs = recs.sort_values('Similarity', ascending=False)

            st.markdown("<h3 class='section-title'>🎯 10 Rekomendasi Serupa</h3>", unsafe_allow_html=True)
            cols = st.columns(2)
            for idx, (_, row) in enumerate(recs.iterrows()):
                with cols[idx % 2]:
                    st.markdown(f"""
                    <div class="anime-card">
                        <div class="title">{row['Name']}</div>
                        <div class="detail">🏷️ {row['Genres']}</div>
                        <div class="detail">⭐ {row['Score']:.2f} · {row['Type']}</div>
                        <div class="detail" style="color:#6c5ce7;">🔗 Similarity: {row['Similarity']:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Table version
            with st.expander("📋 Lihat Tabel Rekomendasi"):
                st.dataframe(recs, use_container_width=True)
        else:
            st.warning("Anime ini tidak memiliki cukup data untuk rekomendasi.")

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.markdown("""
<div class="footer">
    Made with ❤️ using Streamlit • Data from MyAnimeList 2023 • 
    <a href="https://adlanj1.streamlit.app/" target="_blank">Akses Aplikasi</a> • 
    📧 contact@yourdev.com
</div>
""", unsafe_allow_html=True)
