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

    # =================================================
    # METRIC DASHBOARD
    # =================================================
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

    # =================================================
    # DISTRIBUSI KATEGORI
    # =================================================
    st.markdown("""
    <div class="card">
        <h2>📌 Distribusi Kategori Kualitas Udara</h2>
    </div>
    """, unsafe_allow_html=True)

    kategori_count = data["kategori"].value_counts().reset_index()
    kategori_count.columns = ["Kategori", "Jumlah"]

    fig1 = px.pie(
        kategori_count,
        names="Kategori",
        values="Jumlah",
        color="Kategori",
        hole=0.45
    )

    fig1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    st.plotly_chart(fig1, use_container_width=True)

    # =================================================
    # HISTOGRAM PM10
    # =================================================
    st.markdown("""
    <div class="card">
        <h2>🌫️ Distribusi PM10</h2>
    </div>
    """, unsafe_allow_html=True)

    fig2 = px.histogram(
        data,
        x="pm_sepuluh",
        nbins=30,
        color="kategori",
        title="Sebaran Nilai PM10"
    )

    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.08)',
        font_color='white'
    )

    st.plotly_chart(fig2, use_container_width=True)

    # =================================================
    # SCATTER PM10 VS PM2.5
    # =================================================
    st.markdown("""
    <div class="card">
        <h2>📈 Sebaran PM10 dan PM2.5</h2>
    </div>
    """, unsafe_allow_html=True)

    fig3 = px.scatter(
        data,
        x="pm_sepuluh",
        y="pm_duakomalima",
        color="kategori",
        size="pm_duakomalima",
        hover_data=["kategori"],
        title="Hubungan PM10 dan PM2.5"
    )

    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.08)',
        font_color='white'
    )

    st.plotly_chart(fig3, use_container_width=True)

    # =================================================
    # BOXPLOT POLUTAN
    # =================================================
    st.markdown("""
    <div class="card">
        <h2>📦 Boxplot Polutan</h2>
    </div>
    """, unsafe_allow_html=True)

    box_data = data[[
        "pm_sepuluh",
        "pm_duakomalima",
        "sulfur_dioksida",
        "karbon_monoksida",
        "ozon",
        "nitrogen_dioksida"
    ]]

    box_melt = box_data.melt(var_name="Polutan", value_name="Nilai")

    fig4 = px.box(
        box_melt,
        x="Polutan",
        y="Nilai",
        color="Polutan"
    )

    fig4.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.08)',
        font_color='white'
    )

    st.plotly_chart(fig4, use_container_width=True)

    # =================================================
    # HEATMAP KORELASI
    # =================================================
    st.markdown("""
    <div class="card">
        <h2>🔥 Korelasi Antar Polutan</h2>
    </div>
    """, unsafe_allow_html=True)

    corr = data[kolom_numerik].corr()

    fig5 = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues"
    )

    fig5.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    st.plotly_chart(fig5, use_container_width=True)

    # =================================================
    # DATA TABLE
    # =================================================
    st.markdown("""
    <div class="card">
        <h2>📋 Preview Dataset</h2>
    </div>
    """, unsafe_allow_html=True)

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
    kondisi kualitas udara berdasarkan konsentrasi polutan 
    yang terdapat di lingkungan.
    </p>

    <p>
    Website ini memanfaatkan data ISPU (Indeks Standar Pencemar Udara) 
    Provinsi DKI Jakarta tahun 2023 sebagai data pelatihan model.
    </p>

    <br>

    <h2>📌 Variabel yang Digunakan</h2>

    <ul>
        <li><b>PM10</b> → Partikel udara berukuran ≤10 mikrometer</li>
        <li><b>PM2.5</b> → Partikel halus yang dapat masuk ke paru-paru</li>
        <li><b>SO₂</b> → Sulfur Dioksida</li>
        <li><b>CO</b> → Karbon Monoksida</li>
        <li><b>O₃</b> → Ozon</li>
        <li><b>NO₂</b> → Nitrogen Dioksida</li>
    </ul>

    <br>

    <h2>🧠 Cara Kerja Gaussian Naive Bayes</h2>

    <p>
    Gaussian Naive Bayes merupakan algoritma klasifikasi berbasis 
    probabilitas yang menggunakan Teorema Bayes dengan asumsi 
    bahwa setiap variabel independen bersifat saling bebas.
    </p>

    <p>
    Pada metode ini, setiap fitur numerik diasumsikan mengikuti 
    distribusi Gaussian (Normal). Model akan menghitung probabilitas 
    suatu data masuk ke kategori tertentu berdasarkan nilai rata-rata 
    dan standar deviasi masing-masing fitur.
    </p>

    <p>
    Kategori dengan probabilitas tertinggi akan dipilih sebagai 
    hasil prediksi kualitas udara.
    </p>

    <br>

    <h2>🎯 Tujuan Sistem</h2>

    <ul>
        <li>Memprediksi kategori kualitas udara</li>
        <li>Membantu masyarakat memahami kondisi udara</li>
        <li>Memberikan rekomendasi kesehatan berdasarkan hasil prediksi</li>
        <li>Menampilkan visualisasi data polutan secara interaktif</li>
    </ul>

    <br>

    <h2>📊 Dataset</h2>

    <p>
    Dataset yang digunakan berasal dari:
    <br><br>
    Data Indeks Standar Pencemar Udara (ISPU) 
    Provinsi DKI Jakarta Tahun 2023.
    </p>

    </div>
    """, unsafe_allow_html=True)
