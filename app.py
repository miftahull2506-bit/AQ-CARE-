# =====================================================
# MENU NAVIGASI
# =====================================================
selected = option_menu(
    menu_title=None,
    options=["Home", "Prediksi", "Dashboard", "About"],
    icons=["house", "activity", "bar-chart", "info-circle"],
    orientation="horizontal",
)

# =====================================================
# HOME PAGE
# =====================================================
if selected == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">🌍 AQ-CARE</div>
        <div class="hero-sub">
        Smart Air Quality Prediction System <br>
        Using Gaussian Naive Bayes Machine Learning
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    col1.metric("📊 Total Data", len(data))
    col2.metric("🎯 Akurasi Model", f"{akurasi*100:.2f}%")
    col3.metric("🧠 Metode", "Gaussian NB")

    st.write("")

    st.markdown("""
    <div class="card">
        <h2>✨ Tentang AQ-CARE</h2>

        <p>
        AQ-CARE adalah website prediksi kualitas udara berbasis 
        Machine Learning menggunakan metode Gaussian Naive Bayes.
        </p>

        <p>
        Sistem ini dapat membantu pengguna mengetahui kondisi kualitas udara 
        berdasarkan konsentrasi polutan secara cepat dan interaktif.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.image(
        "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
        use_container_width=True
    )

# =====================================================
# HALAMAN PREDIKSI
# =====================================================
elif selected == "Prediksi":

    st.markdown("""
    <div class="card">
        <h2>🌫️ Prediksi Kualitas Udara</h2>
        <p>
        Masukkan nilai polutan udara untuk mengetahui kategori kualitas udara.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        lokasi = st.text_input(
            "📍 Lokasi",
            "Palu, Sulawesi Tengah"
        )

        pm10 = st.number_input("PM10", 0.0, 500.0, 50.0)
        pm25 = st.number_input("PM2.5", 0.0, 500.0, 70.0)
        so2 = st.number_input("SO2", 0.0, 500.0, 30.0)

    with col2:

        co = st.number_input("CO", 0.0, 100.0, 10.0)
        o3 = st.number_input("O3", 0.0, 500.0, 20.0)
        no2 = st.number_input("NO2", 0.0, 500.0, 15.0)

    st.write("")

    prediksi_btn = st.button("🔍 Jalankan Prediksi")

    if prediksi_btn:

        data_input = pd.DataFrame({
            "pm_sepuluh": [pm10],
            "pm_duakomalima": [pm25],
            "sulfur_dioksida": [so2],
            "karbon_monoksida": [co],
            "ozon": [o3],
            "nitrogen_dioksida": [no2]
        })

        hasil = model.predict(data_input)[0]
        probabilitas = model.predict_proba(data_input)[0]

        st.write("")

        if hasil == "BAIK":
            st.success(f"🌿 Kategori Udara: {hasil}")

        elif hasil == "SEDANG":
            st.warning(f"⚠️ Kategori Udara: {hasil}")

        else:
            st.error(f"❌ Kategori Udara: {hasil}")

        kelas = model.classes_

        prob_df = pd.DataFrame({
            "Kategori": kelas,
            "Probabilitas": probabilitas
        })

        fig = px.bar(
            prob_df,
            x="Kategori",
            y="Probabilitas",
            color="Kategori",
            text_auto='.2f',
            title="📊 Probabilitas Prediksi"
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.08)',
            font_color='white'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="rekom">
        <h3>💡 Rekomendasi</h3>
        </div>
        """, unsafe_allow_html=True)

        if hasil == "BAIK":
            st.info("""
            ✅ Udara aman untuk aktivitas luar ruangan  
            ✅ Cocok untuk olahraga  
            ✅ Tidak perlu masker
            """)

        elif hasil == "SEDANG":
            st.warning("""
            ⚠️ Kelompok sensitif disarankan mengurangi aktivitas luar ruangan  
            ⚠️ Gunakan masker bila diperlukan
            """)

        else:
            st.error("""
            ❌ Hindari aktivitas luar ruangan  
            ❌ Gunakan masker  
            ❌ Tutup ventilasi rumah
            """)

# =====================================================
# DASHBOARD PAGE
# =====================================================
elif selected == "Dashboard":

    st.markdown("""
    <div class="card">
        <h1>📊 Air Quality Dashboard</h1>
        <p>
        Dashboard interaktif untuk melihat distribusi kualitas udara,
        pola polutan, dan statistik data ISPU Jakarta 2023.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # METRIC
    total_data = len(data)

    rata_pm10 = round(data["pm_sepuluh"].mean(), 2)
    rata_pm25 = round(data["pm_duakomalima"].mean(), 2)
    rata_o3 = round(data["ozon"].mean(), 2)
    rata_no2 = round(data["nitrogen_dioksida"].mean(), 2)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📊 Total Data", total_data)
    c2.metric("🌫️ Avg PM10", rata_pm10)
    c3.metric("💨 Avg PM2.5", rata_pm25)
    c4.metric("☁️ Avg O₃", rata_o3)

    st.write("")

    # PIE CHART
    kategori_count = data["kategori"].value_counts().reset_index()
    kategori_count.columns = ["Kategori", "Jumlah"]

    fig1 = px.pie(
        kategori_count,
        names="Kategori",
        values="Jumlah",
        hole=0.45,
        color="Kategori"
    )

    fig1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    st.plotly_chart(fig1, use_container_width=True)

    # HISTOGRAM
    fig2 = px.histogram(
        data,
        x="pm_sepuluh",
        color="kategori",
        title="Distribusi PM10"
    )

    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.08)',
        font_color='white'
    )

    st.plotly_chart(fig2, use_container_width=True)

    # SCATTER
    fig3 = px.scatter(
        data,
        x="pm_sepuluh",
        y="pm_duakomalima",
        color="kategori",
        size="pm_duakomalima",
        title="PM10 vs PM2.5"
    )

    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.08)',
        font_color='white'
    )

    st.plotly_chart(fig3, use_container_width=True)

    # DATASET
    st.markdown("## 📋 Preview Dataset")

    st.dataframe(data.head(20), use_container_width=True)

# =====================================================
# ABOUT PAGE
# =====================================================
elif selected == "About":

    st.markdown("""
    <div class="card">

    <h1>ℹ️ About AQ-CARE</h1>

    <p>
    AQ-CARE merupakan sistem prediksi kualitas udara berbasis 
    Machine Learning yang dikembangkan menggunakan metode 
    <b>Gaussian Naive Bayes</b>.
    </p>

    <p>
    Sistem ini bertujuan membantu masyarakat dalam memantau 
    kondisi kualitas udara berdasarkan konsentrasi polutan.
    </p>

    <br>

    <h2>📌 Variabel</h2>

    <ul>
        <li>PM10</li>
        <li>PM2.5</li>
        <li>SO₂</li>
        <li>CO</li>
        <li>O₃</li>
        <li>NO₂</li>
    </ul>

    <br>

    <h2>🧠 Cara Kerja Gaussian Naive Bayes</h2>

    <p>
    Gaussian Naive Bayes merupakan algoritma klasifikasi berbasis probabilitas 
    menggunakan Teorema Bayes dengan asumsi distribusi normal.
    </p>

    <p>
    Model akan menghitung probabilitas setiap kategori kualitas udara 
    berdasarkan nilai polutan yang dimasukkan pengguna.
    </p>

    <br>

    <h2>🎯 Tujuan Sistem</h2>

    <ul>
        <li>Memprediksi kualitas udara</li>
        <li>Menyediakan visualisasi data polutan</li>
        <li>Membantu masyarakat memahami kondisi udara</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
AQ-CARE © 2026 <br>
Gaussian Naive Bayes Air Quality Prediction System
</div>
""", unsafe_allow_html=True)
