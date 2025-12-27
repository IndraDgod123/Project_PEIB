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
    "📋 Data Cluster",
    "📈 Visualisasi PCA",
    "🧠 Interpretasi Cluster",
    "🎯 Simulasi Pelanggan"
    ])

    # ===============================
    # TAB 1 — DATA
    # ===============================
    with tab1:
        st.subheader("Hasil Clustering Customer")
        st.dataframe(rfm)

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