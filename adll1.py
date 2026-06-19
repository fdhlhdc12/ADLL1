def show_topnav():
    avatar_html = f'<img src="data:image/png;base64,{AVATAR_IMAGE}">' if AVATAR_IMAGE else "🧑‍🚀"
    moon_icon = "☀️" if not st.session_state.dark_mode else "🌙"

    # Buat 2 kolom: kiri kosong, kanan berisi tombol + user
    col_left, col_right = st.columns([6, 1.5])
    with col_left:
        pass  # kosong, biar elemen di kanan
    with col_right:
        # Dua subkolom tanpa gap
        sub1, sub2 = st.columns([1, 3], gap="small")
        with sub1:
            # Tombol tema
            if st.button(moon_icon, key="theme_toggle", help="Toggle Dark/Light Theme", use_container_width=True):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()
            # Styling tombol bulat
            st.markdown("""
            <style>
            div[data-testid="column"]:nth-of-type(2) div[data-testid="column"]:nth-of-type(1) button {
                background: rgba(255,255,255,0.06) !important;
                border: 1px solid rgba(255,255,255,0.08) !important;
                border-radius: 50% !important;
                width: 40px !important;
                height: 40px !important;
                font-size: 18px !important;
                padding: 0 !important;
                color: #e2e8f0 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                min-width: 40px !important;
            }
            div[data-testid="column"]:nth-of-type(2) div[data-testid="column"]:nth-of-type(1) button:hover {
                background: rgba(255,255,255,0.12) !important;
            }
            </style>
            """, unsafe_allow_html=True)
        with sub2:
            # Profil user
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); padding:4px 12px 4px 4px; border-radius:999px; margin-left:2px;">
                <div style="width:32px; height:32px; border-radius:50%; background:linear-gradient(135deg,#7c3aed,#ec4899); display:flex; align-items:center; justify-content:center; font-size:16px; overflow:hidden;">
                    {avatar_html}
                </div>
                <div style="font-size:14px; font-weight:600; color:white;">Otaku User</div>
            </div>
            """, unsafe_allow_html=True)
