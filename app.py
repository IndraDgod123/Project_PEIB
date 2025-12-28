import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    with open("rfm_kmeans_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
scaler = model["scaler"]
kmeans = model["kmeans"]
cluster_profile = model["cluster_profile"]

cluster_interpretation = {
    0: {
        "name": "Loyal High-Value Customers",
        "desc": "Pelanggan paling loyal dengan frekuensi dan nilai transaksi tinggi.",
        "strategy": "Pertahankan dengan program loyalitas dan layanan premium."
    },
    1: {
        "name": "Big Spenders but Infrequent",
        "desc": "Pelanggan dengan nilai belanja tinggi namun jarang bertransaksi.",
        "strategy": "Dorong frekuensi dengan promo personal dan reminder."
    },
    2: {
        "name": "Occasional Low-Value Customers",
        "desc": "Pelanggan dengan aktivitas dan nilai transaksi rendah.",
        "strategy": "Edukasi produk dan promo ringan."
    },
    3: {
        "name": "Recent Low Spenders",
        "desc": "Pelanggan aktif terbaru namun dengan nilai transaksi kecil.",
        "strategy": "Lakukan upselling dan cross-selling."
    },
    4: {
        "name": "Dormant Big Spenders",
        "desc": "Pelanggan bernilai tinggi namun sudah lama tidak aktif.",
        "strategy": "Lakukan win-back campaign."
    },
    5: {
        "name": "Recent Big Spenders",
        "desc": "Pelanggan dengan transaksi terbaru dan nilai belanja sangat tinggi.",
        "strategy": "Berikan layanan personal dan penawaran eksklusif."
    },
    6: {
        "name": "Frequent High-Value Customers",
        "desc": "Pelanggan dengan frekuensi dan nilai transaksi tinggi.",
        "strategy": "Bangun loyalitas jangka panjang (membership)."
    },
    7: {
        "name": "At-Risk High-Value Customers",
        "desc": "Pelanggan bernilai tinggi namun mulai jarang bertransaksi.",
        "strategy": "Cegah churn dengan pendekatan personal."
    }
}
# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="RFM Clustering Dashboard", layout="wide")

st.title("📊 RFM Customer Clustering Dashboard")
st.markdown("Upload dataset CSV untuk melihat hasil clustering & visualisasi.")

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # ===============================
    # PREPROCESSING (SAMA DENGAN MODEL)
    # ===============================
    df = df.rename(columns={
        "Purchase Amount (USD)": "Purchase_Amount",
        "Frequency of Purchases": "Frequency_of_Purchases",
        "Previous Purchases": "Previous_Purchases"
    })

    def recency_score(x):
        if x <= 10: return 1
        elif x <= 20: return 2
        elif x <= 30: return 3
        elif x <= 40: return 4
        else: return 5

    def frequency_score(x):
        if x == "Weekly":
            return 5
        elif x in ["Bi-Weekly", "Fortnightly"]:
            return 4
        elif x == "Monthly":
            return 3
        elif x in ["Quarterly", "Every 3 Months"]:
            return 2
        else:
            return 1

    def monetary_score(x):
        if x > 40: return 5
        elif x > 30: return 4
        elif x > 20: return 3
        elif x > 10: return 2
        else: return 1

    df["R"] = df["Previous_Purchases"].apply(recency_score)
    df["F"] = df["Frequency_of_Purchases"].apply(frequency_score)
    df["M"] = df["Purchase_Amount"].apply(monetary_score)

    # ===============================
    # AGREGASI PER CUSTOMER
    # ===============================
    rfm = df.groupby("Customer ID")[["R","F","M"]].max().reset_index()
    rfm["RFM_score"] = (rfm["R"] + rfm["F"] + rfm["M"]) / 3

    # ===============================
    # NORMALISASI (HANYA R,F,M)
    # ===============================
    X_scaled = scaler.transform(rfm[["R","F","M"]])

    # ===============================
    # CLUSTERING
    # ===============================
    rfm["Cluster"] = kmeans.predict(X_scaled)

    # ===============================
    # PCA VISUALIZATION
    # ===============================
    pca = PCA(n_components=2)
    pca_vals = pca.fit_transform(X_scaled)
    rfm["PCA1"] = pca_vals[:,0]
    rfm["PCA2"] = pca_vals[:,1]

    # ===============================
    # TABS
    # ===============================
    tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data & EDA",
    "📈 Visualisasi PCA",
    "🧠 Interpretasi Cluster",
    "🎯 Simulasi Pelanggan"
    ])

    with tab1:
        st.subheader("📋 Data & Exploratory Data Analysis (EDA)")

    # ===============================
    # FILTER CLUSTER
    # ===============================
        st.markdown("### 🔎 Filter Cluster")

        selected_clusters = st.multiselect(
            "Pilih cluster yang ingin dianalisis:",
            options=sorted(rfm["Cluster"].unique()),
            default=sorted(rfm["Cluster"].unique())
        )

        filtered_rfm = rfm[rfm["Cluster"].isin(selected_clusters)]

    # ===============================
    # DATA PREVIEW
    # ===============================
        st.markdown("### 📄 Preview Data Hasil Clustering")
        st.dataframe(filtered_rfm, use_container_width=True)

    # ===============================
    # METRIC RINGKAS
    # ===============================
        st.markdown("### 📊 Ringkasan Statistik")

        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Customer", len(filtered_rfm))
        col2.metric("Jumlah Cluster Aktif", filtered_rfm["Cluster"].nunique())
        col3.metric("Rata-rata Skor RFM", round(filtered_rfm[["R","F","M"]].mean().mean(), 2))

        st.markdown("---")

    # ===============================
    # PIE CHART PROPORSI CLUSTER
    # ===============================
        st.markdown("### 🥧 Proporsi Customer per Cluster")
        st.caption(
        "Visualisasi ini menunjukkan distribusi jumlah pelanggan pada setiap cluster. "
        "Cluster dengan proporsi terbesar menandakan segmen dominan dalam basis pelanggan."
        )   
        cluster_counts = filtered_rfm["Cluster"].value_counts().sort_index()

        fig1, ax1 = plt.subplots(figsize=(5,5))
        ax1.pie(
            cluster_counts,
            labels=[f"Cluster {i}" for i in cluster_counts.index],
            autopct="%1.1f%%",
            startangle=90
        )
        ax1.set_title("Distribusi Customer per Cluster")
        st.pyplot(fig1)

        # Interpretasi Pie
        dominant_cluster = cluster_counts.idxmax()
        dominant_pct = (cluster_counts.max() / cluster_counts.sum()) * 100

        st.markdown(
            f"🧠 **Cluster {dominant_cluster}** merupakan cluster dominan "
            f"dengan proporsi sekitar **{dominant_pct:.1f}%** dari total customer terpilih."
        )

        st.markdown("---")

    # ===============================
    # HEATMAP KORELASI
    # ===============================
        st.markdown("### 🔥 Korelasi RFM terhadap Purchase Amount")
        st.caption(
        "Heatmap ini menunjukkan hubungan antara variabel R, F, M dengan "
        "Purchase Amount. Warna merah menunjukkan korelasi positif kuat, "
        "sedangkan biru menunjukkan korelasi negatif."
        )

        heatmap_df = df.merge(
            filtered_rfm[["Customer ID", "Cluster"]],
            on="Customer ID",
            how="inner"
        )

        corr_data = heatmap_df[["R","F","M","Purchase_Amount"]].corr()

        fig2, ax2 = plt.subplots(figsize=(5.5,4.5))
        im = ax2.imshow(corr_data, cmap="coolwarm")

        ax2.set_xticks(range(len(corr_data.columns)))
        ax2.set_yticks(range(len(corr_data.columns)))
        ax2.set_xticklabels(corr_data.columns)
        ax2.set_yticklabels(corr_data.columns)

        plt.colorbar(im, ax=ax2, fraction=0.046)
        ax2.set_title("Heatmap Korelasi")
        st.pyplot(fig2)

    # Interpretasi Heatmap
        r_corr = corr_data.loc["R", "Purchase_Amount"]
        f_corr = corr_data.loc["F", "Purchase_Amount"]
        m_corr = corr_data.loc["M", "Purchase_Amount"]

        st.markdown("#### 🧠 Interpretasi Korelasi")

        if abs(m_corr) > abs(r_corr) and abs(m_corr) > abs(f_corr):
            st.markdown(
                f"- **Monetary (M)** memiliki korelasi paling kuat "
                f"dengan Purchase Amount (r = {m_corr:.2f}), "
                f"menunjukkan bahwa nilai belanja sangat dipengaruhi oleh aspek monetary."
            )
        elif abs(f_corr) > abs(r_corr):
            st.markdown(
                f"- **Frequency (F)** memiliki pengaruh lebih besar "
                f"terhadap nilai pembelian (r = {f_corr:.2f})."
            )
        else:
            st.markdown(
                f"- **Recency (R)** menunjukkan hubungan yang relatif lebih dominan "
                f"dibandingkan F dan M (r = {r_corr:.2f})."
            )

        st.markdown("---")

    # ===============================
    # BOXPLOT PURCHASE AMOUNT
    # ===============================
        st.markdown("### 📦 Distribusi Purchase Amount per Cluster")

        fig3, ax3 = plt.subplots(figsize=(6.5,4.5))
        heatmap_df.boxplot(
            column="Purchase_Amount",
            by="Cluster",
            ax=ax3
        )
        ax3.set_title("Distribusi Nilai Pembelian")
        ax3.set_xlabel("Cluster")
        ax3.set_ylabel("Purchase Amount (USD)")
        plt.suptitle("")

        st.pyplot(fig3)

    # Interpretasi Boxplot
        median_purchase = heatmap_df.groupby("Cluster")["Purchase_Amount"].median()
        top_cluster = median_purchase.idxmax()

        st.markdown(
            f"🧠 **Cluster {top_cluster}** memiliki median Purchase Amount tertinggi, "
            f"yang mengindikasikan kelompok pelanggan dengan potensi pendapatan terbesar."
        )

        st.info(
            "📌 Tab ini digunakan untuk memahami karakteristik data, distribusi cluster, "
            "serta hubungan antara variabel RFM dan nilai transaksi sebelum analisis lanjutan."
        )

    # ===============================
    # TAB 2 — PCA
    # ===============================
    with tab2:
        st.subheader("Visualisasi PCA (2D)")

        fig, ax = plt.subplots(figsize=(8,6))
        scatter = ax.scatter(
            rfm["PCA1"], rfm["PCA2"],
            c=rfm["Cluster"], cmap="tab10", alpha=0.7
        )
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("PCA Clustering Visualization")
        plt.colorbar(scatter, ax=ax, label="Cluster")

        st.pyplot(fig)

    # ===============================
    # TAB 3 — INTERPRETASI
    # ===============================
    with tab3:
        st.subheader("🧠 Interpretasi Cluster")

        for cluster_id in sorted(cluster_interpretation.keys()):
            info = cluster_interpretation[cluster_id]
            profile = cluster_profile.loc[cluster_id]

            with st.expander(f"Cluster {cluster_id} — {info['name']}"):
                st.markdown(f"**Deskripsi:** {info['desc']}")
                st.markdown(f"**Strategi:** {info['strategy']}")

                st.markdown("**Profil RFM (rata-rata):**")
                st.dataframe(profile.to_frame("Nilai"))
# ===============================
# TAB 4 — SIMULASI RFM (FINAL)
# ===============================
    with tab4:
        st.subheader("🎯 Simulasi Pelanggan (Input R, F, M)")

        col1, col2, col3 = st.columns(3)
        with col1:
            r_input = st.slider("Recency (R)", 1, 5, 3)
        with col2:
            f_input = st.slider("Frequency (F)", 1, 5, 3)
        with col3:
            m_input = st.slider("Monetary (M)", 1, 5, 3)

        if st.button("🔍 Prediksi Cluster"):
        # -------------------------------
        # PREDIKSI CLUSTER
        # -------------------------------
            input_df = pd.DataFrame([[r_input, f_input, m_input]], columns=["R","F","M"])
            input_scaled = scaler.transform(input_df)
            cluster_pred = int(kmeans.predict(input_scaled)[0])

            info = cluster_interpretation[cluster_pred]
            profile = cluster_profile.loc[cluster_pred]

            st.success(f"✅ Pelanggan masuk ke **Cluster {cluster_pred} — {info['name']}**")

        # -------------------------------
        # INTERPRETASI RESMI CLUSTER
        # -------------------------------
            st.markdown("### 🧠 Interpretasi Cluster")
            st.markdown(f"""
            **Nama Cluster:** {info['name']}  
            **Deskripsi:** {info['desc']}  
            **Strategi Bisnis:** {info['strategy']}
            """)

        # -------------------------------
        # PROFIL RFM CLUSTER
        # -------------------------------
            st.markdown("### 📊 Profil RFM Rata-rata Cluster")
            st.dataframe(profile.to_frame("Nilai"))

        # -------------------------------
        # RADAR CHART INPUT USER
        # -------------------------------
            st.markdown("### 📈 Profil RFM Input Pelanggan")

            labels = ["R", "F", "M"]
            values = [r_input, f_input, m_input]
            values += values[:1]
            angles = np.linspace(0, 2*np.pi, len(labels)+1)

            fig, ax = plt.subplots(subplot_kw=dict(polar=True))
            ax.plot(angles, values, linewidth=2)
            ax.fill(angles, values, alpha=0.3)
            ax.set_thetagrids(np.degrees(angles[:-1]), labels)
            ax.set_ylim(0, 5)
            ax.set_title("Profil RFM Pelanggan")

            st.pyplot(fig)

        # -------------------------------
        # POSISI PELANGGAN PADA PCA
        # -------------------------------
            st.markdown("### 📍 Posisi Pelanggan dalam Peta Cluster (PCA)")

            input_pca = pca.transform(input_scaled)

            fig2, ax2 = plt.subplots(figsize=(7,5))
            ax2.scatter(
                rfm["PCA1"], rfm["PCA2"],
                c=rfm["Cluster"], cmap="tab10", alpha=0.4
            )
            ax2.scatter(
                input_pca[0,0], input_pca[0,1],
                color="red", s=150, label="Pelanggan"
            )
            ax2.legend()
            ax2.set_xlabel("PCA 1")
            ax2.set_ylabel("PCA 2")
            ax2.set_title("Posisi Pelanggan terhadap Cluster")

            st.pyplot(fig2)
else:
    st.info("Silakan upload file CSV terlebih dahulu.")