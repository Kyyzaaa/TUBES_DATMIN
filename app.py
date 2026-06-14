"""
app.py  —  Hotel Booking Demand  |  Data Mining Dashboard
Dua halaman:
  1. Prediksi   → input manual → prediksi cluster + regression score
  2. Visualisasi → EDA, clustering, regresi, klasifikasi + interpretasi
"""

import os, json, warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Booking — Data Mining",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "adr", "lead_time", "total_stays", "total_guests",
    "total_of_special_requests", "previous_cancellations",
]
LABEL_MAP   = {0: "🟦 Cluster 0 — Budget / Short-stay",
               1: "🟠 Cluster 1 — Premium / Long-stay"}
COLOR_MAP   = {0: "#2196F3", 1: "#FF9800"}
MODEL_DIR   = "models"

# ──────────────────────────────────────────────────────────────────────────────
# LOAD ARTEFAK (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model…")
def load_artifacts():
    a = {}
    a["scaler_cluster"] = joblib.load(f"{MODEL_DIR}/scaler_cluster.pkl")
    a["kmeans"]         = joblib.load(f"{MODEL_DIR}/kmeans.pkl")
    a["cluster_map"]    = joblib.load(f"{MODEL_DIR}/cluster_map.pkl")

    a["scaler_reg"]  = joblib.load(f"{MODEL_DIR}/scaler_reg.pkl")
    a["lr"]          = joblib.load(f"{MODEL_DIR}/linear_regression.pkl")
    a["rfr"]         = joblib.load(f"{MODEL_DIR}/rf_regressor.pkl")

    a["scaler_clf"]  = joblib.load(f"{MODEL_DIR}/scaler_clf.pkl")
    a["log_reg"]     = joblib.load(f"{MODEL_DIR}/logistic_regression.pkl")
    a["nb"]          = joblib.load(f"{MODEL_DIR}/naive_bayes.pkl")
    a["rfc"]         = joblib.load(f"{MODEL_DIR}/rf_classifier.pkl")
    return a

@st.cache_data(show_spinner="Memuat data…")
def load_data():
    df = pd.read_parquet(f"{MODEL_DIR}/df_clean.parquet")
    with open(f"{MODEL_DIR}/meta.json") as f:
        meta = json.load(f)
    return df, meta

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hotel.png", width=72)
    st.title("🏨 Hotel Booking\nData Mining")
    st.markdown("---")
    page = st.radio(
        "Navigasi Halaman",
        ["🔮 Prediksi", "📊 Visualisasi & Interpretasi"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Dataset: Hotel Booking Demand\nModel: K-Means K=2, Regresi, Klasifikasi")

# ──────────────────────────────────────────────────────────────────────────────
# CHECK MODEL FILES
# ──────────────────────────────────────────────────────────────────────────────
if not os.path.exists(f"{MODEL_DIR}/kmeans.pkl"):
    st.error("❌ Model belum dilatih. Jalankan `python train_models.py` terlebih dahulu.")
    st.stop()

art  = load_artifacts()
df, meta = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 1 — PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Prediksi":

    st.title("🔮 Prediksi Segmentasi Tamu Hotel")
    st.markdown(
        "Masukkan data booking untuk memprediksi **segmen tamu** (Cluster), "
        "nilai regresi, dan probabilitas klasifikasi."
    )

    # ── INPUT FORM ────────────────────────────────────────────────────────────
    st.subheader("📝 Input Data Booking")

    col1, col2, col3 = st.columns(3)

    with col1:
        adr = st.number_input(
            "💰 ADR (Average Daily Rate)",
            min_value=0.0, max_value=600.0, value=95.0, step=5.0,
            help="Tarif rata-rata per malam (USD/EUR)",
        )
        lead_time = st.number_input(
            "📅 Lead Time (hari)",
            min_value=0, max_value=700, value=60,
            help="Jumlah hari antara booking dan check-in",
        )

    with col2:
        total_stays = st.number_input(
            "🌙 Total Stays (malam)",
            min_value=0, max_value=60, value=3,
            help="Total malam menginap (weekend + weekday)",
        )
        total_guests = st.number_input(
            "👥 Total Guests",
            min_value=1, max_value=20, value=2,
            help="Jumlah total tamu (dewasa + anak + bayi)",
        )

    with col3:
        special_req = st.number_input(
            "⭐ Special Requests",
            min_value=0, max_value=10, value=0,
            help="Jumlah permintaan khusus dari tamu",
        )
        prev_cancel = st.number_input(
            "❌ Previous Cancellations",
            min_value=0, max_value=30, value=0,
            help="Riwayat pembatalan sebelumnya",
        )

    st.markdown("---")

    # ── PREDIKSI ──────────────────────────────────────────────────────────────
    if st.button("🚀 Prediksi Sekarang", type="primary", use_container_width=True):

        input_arr = np.array([[adr, lead_time, total_stays, total_guests, special_req, prev_cancel]])

        # Clustering
        X_sc        = art["scaler_cluster"].transform(input_arr)
        raw_label   = art["kmeans"].predict(X_sc)[0]
        cluster_map = art["cluster_map"]
        stable_label = int(cluster_map.tolist().index(raw_label))

        # Regression
        X_rsc    = art["scaler_reg"].transform(input_arr)
        reg_lr   = float(art["lr"].predict(X_rsc)[0])
        reg_rfr  = float(art["rfr"].predict(X_rsc)[0])

        # Classification
        X_csc    = art["scaler_clf"].transform(input_arr)
        prob_lr  = art["log_reg"].predict_proba(X_csc)[0]
        prob_nb  = art["nb"].predict_proba(X_csc)[0]
        prob_rfc = art["rfc"].predict_proba(X_csc)[0]
        clf_lr   = int(art["log_reg"].predict(X_csc)[0])

        # ── HASIL ─────────────────────────────────────────────────────────────
        st.markdown("## 📊 Hasil Prediksi")

        # Cluster result card
        bg_col  = "#E3F2FD" if stable_label == 0 else "#FFF3E0"
        bd_col  = COLOR_MAP[stable_label]
        segment = "Budget / Short-stay" if stable_label == 0 else "Premium / Long-stay"
        icon    = "🟦" if stable_label == 0 else "🟠"

        st.markdown(f"""
        <div style="background:{bg_col}; border-left:6px solid {bd_col};
                    padding:20px 24px; border-radius:8px; margin-bottom:16px;">
            <h2 style="margin:0; color:{bd_col};">{icon} Cluster {stable_label} — {segment}</h2>
            <p style="margin:6px 0 0 0; color:#555;">
                Tamu diprediksi masuk segmen <strong>{segment}</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Metric cards
        c1, c2, c3 = st.columns(3)
        c1.metric("Regresi — Linear", f"{reg_lr:.3f}",
                  help="Nilai prediksi kontinu label cluster (Linear Regression)")
        c2.metric("Regresi — RF", f"{reg_rfr:.3f}",
                  help="Nilai prediksi kontinu label cluster (Random Forest Regressor)")
        c3.metric("Klasifikasi", f"Cluster {clf_lr}",
                  help="Hasil prediksi diskrit (Logistic Regression)")

        st.markdown("---")

        # Probabilitas
        st.subheader("📈 Probabilitas Klasifikasi")
        prob_df = pd.DataFrame({
            "Model": ["Logistic Regression", "Naive Bayes", "Random Forest"],
            "P(Cluster 0 — Budget)":  [prob_lr[0], prob_nb[0], prob_rfc[0]],
            "P(Cluster 1 — Premium)": [prob_lr[1], prob_nb[1], prob_rfc[1]],
        })

        fig, ax = plt.subplots(figsize=(9, 3))
        x = np.arange(3)
        width = 0.35
        ax.bar(x - width/2, prob_df["P(Cluster 0 — Budget)"],  width, label="Cluster 0", color="#2196F3", alpha=0.85)
        ax.bar(x + width/2, prob_df["P(Cluster 1 — Premium)"], width, label="Cluster 1", color="#FF9800", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(prob_df["Model"])
        ax.set_ylabel("Probabilitas"); ax.set_ylim(0, 1.1)
        ax.legend(); ax.set_title("Probabilitas per Model Klasifikasi")
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Tabel probabilitas
        st.dataframe(
            prob_df.set_index("Model").style.format("{:.4f}")
                   .background_gradient(cmap="Blues", subset=["P(Cluster 0 — Budget)"])
                   .background_gradient(cmap="Oranges", subset=["P(Cluster 1 — Premium)"]),
            use_container_width=True,
        )

        # Interpretasi
        st.markdown("---")
        st.subheader("💡 Interpretasi")

        if stable_label == 0:
            st.info("""
**Cluster 0 — Budget / Short-stay**

Tamu dengan karakteristik ini cenderung:
- Melakukan pemesanan mendadak (lead time pendek)
- Menginap dalam waktu singkat (≤ 3 malam)
- Memilih kamar dengan harga terjangkau (ADR rendah)
- Tamu perorangan atau pasangan (total guests sedikit)

**Rekomendasi bisnis:** Tawarkan paket last-minute deals atau promo malam ini.
            """)
        else:
            st.warning("""
**Cluster 1 — Premium / Long-stay**

Tamu dengan karakteristik ini cenderung:
- Memesan jauh hari sebelumnya (lead time panjang)
- Menginap lebih lama (≥ 4 malam)
- Bersedia membayar tarif premium (ADR tinggi)
- Membawa lebih banyak tamu atau keluarga

**Rekomendasi bisnis:** Tawarkan paket menginap extended, upgrade kamar, dan layanan concierge.
            """)

# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 2 — VISUALISASI & INTERPRETASI
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("📊 Visualisasi & Interpretasi")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔵 Clustering", "🟡 Regresi", "🔴 Klasifikasi", "📋 Ringkasan"
    ])

    # ── TAB 1: CLUSTERING ──────────────────────────────────────────────────────
    with tab1:
        st.subheader("K-Means Clustering (K=2)")

        # Metrics
        c0_n = int(meta["cluster_counts"]["0"])
        c1_n = int(meta["cluster_counts"]["1"])
        total_n = c0_n + c1_n
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Silhouette Score",  f"{meta['silhouette']:.4f}")
        m2.metric("Inertia",           f"{meta['inertia']:,.0f}")
        m3.metric("Cluster 0 (Budget)",  f"{c0_n:,}  ({c0_n/total_n*100:.1f}%)")
        m4.metric("Cluster 1 (Premium)", f"{c1_n:,}  ({c1_n/total_n*100:.1f}%)")

        st.markdown("---")

        # ── PCA Scatter + Bar distribution ────────────────────────────────────
        st.markdown("#### 🗺️ Visualisasi Cluster (PCA 2D) & Distribusi")

        X_cluster = df[FEATURES].copy()
        scaler_c  = art["scaler_cluster"]
        kmeans_m  = art["kmeans"]
        pca_m     = PCA(n_components=2, random_state=42)

        # Subsample for speed
        sample_idx = np.random.RandomState(42).choice(len(df), min(10000, len(df)), replace=False)
        X_sub = scaler_c.transform(X_cluster.iloc[sample_idx])
        X_pca = pca_m.fit_transform(X_sub)
        labels_sub = df["cluster"].values[sample_idx]

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for cl in [0, 1]:
            mask = labels_sub == cl
            axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                            c=COLOR_MAP[cl], alpha=0.3, s=6, label=f"Cluster {cl}")
        centroids_pca = pca_m.transform(scaler_c.transform(
            scaler_c.inverse_transform(kmeans_m.cluster_centers_)
        ))
        axes[0].scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                        c="red", s=200, marker="X", zorder=5, label="Centroid")
        var_exp = pca_m.explained_variance_ratio_
        axes[0].set_title(f"Visualisasi Cluster K=2 (PCA 2D)\nVariance explained: {sum(var_exp)*100:.1f}%")
        axes[0].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
        axes[0].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
        axes[0].legend()

        counts = [c0_n, c1_n]
        axes[1].bar(["Cluster 0", "Cluster 1"], counts, color=[COLOR_MAP[0], COLOR_MAP[1]], edgecolor="black", width=0.5)
        axes[1].set_title("Jumlah Data per Cluster")
        axes[1].set_ylabel("Jumlah Booking")
        for i, v in enumerate(counts):
            axes[1].text(i, v + 200, f"{v:,}\n({v/total_n*100:.1f}%)", ha="center", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""
**Interpretasi PCA Scatter:**
Kedua cluster terpisah dengan jelas di ruang 2 dimensi hasil PCA. Cluster 0 (biru) mendominasi area dengan nilai PC1 rendah — menunjukkan tamu dengan lead time pendek dan ADR rendah. Cluster 1 (oranye) berada di area PC1 tinggi, mencerminkan tamu premium dengan pemesanan jauh hari sebelumnya.
        """)

        # ── Boxplot per fitur ──────────────────────────────────────────────────
        st.markdown("#### 📦 Distribusi Fitur per Cluster")

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Perbandingan Distribusi Fitur per Cluster", fontsize=13, fontweight="bold")
        for ax, col in zip(axes.flatten(), FEATURES):
            data_plot = df[[col, "cluster"]].copy()
            data_plot[col] = data_plot[col].clip(upper=data_plot[col].quantile(0.99))
            sns.boxplot(x="cluster", y=col, data=data_plot,
                        palette=[COLOR_MAP[0], COLOR_MAP[1]], ax=ax)
            ax.set_title(col); ax.set_xlabel("Cluster")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""
**Interpretasi Boxplot:**
Cluster 1 secara konsisten menunjukkan nilai lebih tinggi pada ADR, lead time, total stays, dan total guests — mengonfirmasi karakteristik tamu Premium/Long-stay. Cluster 0 memiliki variasi lebih kecil dan nilai median yang lebih rendah, sesuai profil Budget/Short-stay.
        """)

        # ── Centroid table ─────────────────────────────────────────────────────
        st.markdown("#### 📍 Nilai Centroid Cluster (Skala Asli)")
        centroid_orig = scaler_c.inverse_transform(kmeans_m.cluster_centers_)
        cmap_arr = art["cluster_map"]
        # Reorder centroid by stable label
        centroid_stable = centroid_orig[np.argsort(cmap_arr)]
        centroid_df = pd.DataFrame(
            centroid_stable, columns=FEATURES,
            index=["Cluster 0 (Budget)", "Cluster 1 (Premium)"]
        ).round(2)
        st.dataframe(centroid_df.style.background_gradient(cmap="RdYlGn", axis=0),
                     use_container_width=True)

    # ── TAB 2: REGRESI ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Regresi (Target = Label Cluster)")

        st.markdown("""
Label cluster (0/1) digunakan sebagai target regresi kontinu untuk menguji kemampuan model dalam **memprediksi nilai cluster secara numerik**.
        """)

        # Metrics table
        reg_data = []
        for model_name, vals in meta["reg_metrics"].items():
            reg_data.append({"Model": model_name, **vals})
        reg_df = pd.DataFrame(reg_data)
        best_reg = reg_df.loc[reg_df["R2"].idxmax(), "Model"]

        st.markdown("#### 📋 Hasil Evaluasi Model Regresi")
        st.dataframe(
            reg_df.set_index("Model").style
                  .highlight_max(subset=["R2"], color="#c6efce")
                  .highlight_min(subset=["MAE","RMSE"], color="#c6efce")
                  .format("{:.4f}"),
            use_container_width=True,
        )
        st.success(f"✅ **Model Terbaik: {best_reg}**  —  berdasarkan R² tertinggi dan RMSE terkecil.")

        # Bar chart perbandingan
        st.markdown("#### 📊 Perbandingan Metrik Regresi")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        metrics_plot = ["MAE", "RMSE", "R2"]
        labels_ = [m.replace("_", " ") for m in reg_df["Model"]]
        colors_ = ["#42A5F5", "#FF7043"]
        for ax, met in zip(axes, metrics_plot):
            vals_p = reg_df[met].values
            bars = ax.bar(labels_, vals_p, color=colors_, edgecolor="black")
            ax.set_title(met)
            ax.set_ylabel(met)
            for bar, v in zip(bars, vals_p):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                        f"{v:.4f}", ha="center", fontsize=9)
            plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""
**Interpretasi:**
Random Forest Regressor unggul jauh dibanding Linear Regression (R² ≈ 0.966 vs 0.635). Ini menunjukkan bahwa hubungan antara fitur booking dengan label cluster bersifat **non-linear** — sesuatu yang tidak bisa ditangkap oleh model linear sederhana. Lead time dan ADR menjadi prediktor utama perbedaan antar cluster.
        """)

        # Feature importance
        st.markdown("#### 🔍 Feature Importance — Random Forest Regressor")
        rfr_model = art["rfr"]
        fi = pd.Series(rfr_model.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        fi.plot(kind="barh", ax=ax, color="coral", edgecolor="black")
        ax.set_title("Feature Importance — Random Forest Regressor")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""
**Lead Time** adalah fitur paling berpengaruh dalam membedakan cluster, diikuti **ADR**. Artinya, perilaku pemesanan (seberapa jauh hari sebelumnya tamu memesan) lebih menentukan segmen tamu dibandingkan harga kamar itu sendiri.
        """)

    # ── TAB 3: KLASIFIKASI ─────────────────────────────────────────────────────
    with tab3:
        st.subheader("Klasifikasi (Target = Label Cluster)")

        st.markdown("""
Tiga model klasifikasi dilatih untuk memprediksi label cluster secara diskrit: **Logistic Regression**, **Naive Bayes**, dan **Random Forest**.
        """)

        # Metrics table
        clf_data = []
        for model_name, vals in meta["clf_metrics"].items():
            clf_data.append({"Model": model_name, **vals})
        clf_df_display = pd.DataFrame(clf_data)
        best_clf = clf_df_display.loc[clf_df_display["F1"].idxmax(), "Model"]

        st.markdown("#### 📋 Hasil Evaluasi Model Klasifikasi")
        st.dataframe(
            clf_df_display.set_index("Model").style
                          .highlight_max(color="#c6efce")
                          .format("{:.4f}"),
            use_container_width=True,
        )
        st.success(f"✅ **Model Terbaik: {best_clf}**  —  berdasarkan F1-Score tertinggi.")

        # Radar / bar comparison
        st.markdown("#### 📊 Perbandingan Metrik Klasifikasi")
        metrics_clf = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        fig, ax = plt.subplots(figsize=(12, 4))
        x_pos = np.arange(len(metrics_clf))
        width = 0.25
        colors_m = ["#42A5F5", "#EF5350", "#66BB6A"]
        for i, (_, row) in enumerate(clf_df_display.iterrows()):
            vals_m = [row["Accuracy"], row["Precision"], row["Recall"], row["F1"], row["AUC"]]
            bars = ax.bar(x_pos + i*width, vals_m, width, label=row["Model"],
                          color=colors_m[i], alpha=0.85, edgecolor="black")
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(metrics_clf)
        ax.set_ylim(0.8, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Perbandingan Metrik Klasifikasi")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # ROC Curve
        st.markdown("#### 📈 ROC Curve")

        # Subsample test set
        from sklearn.model_selection import train_test_split
        y_clf_full = df["cluster"].astype(int)
        X_full = df[FEATURES]
        _, X_te_c, _, y_te_c = train_test_split(
            X_full, y_clf_full, test_size=0.2, random_state=42, stratify=y_clf_full
        )
        X_te_cs = art["scaler_clf"].transform(X_te_c)

        fig, ax = plt.subplots(figsize=(8, 5))
        model_roc = {
            "Logistic Regression": art["log_reg"],
            "Naive Bayes":         art["nb"],
            "Random Forest":       art["rfc"],
        }
        for name, model in model_roc.items():
            y_prob = model.predict_proba(X_te_cs)[:, 1]
            from sklearn.metrics import roc_curve, roc_auc_score
            fpr, tpr, _ = roc_curve(y_te_c, y_prob)
            auc_val = roc_auc_score(y_te_c, y_prob)
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_val:.3f})")
        ax.plot([0,1],[0,1],"k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Semua Model Klasifikasi")
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""
**Interpretasi ROC Curve:**
Logistic Regression dan Random Forest mencapai AUC mendekati **1.000**, menunjukkan kemampuan diskriminasi yang hampir sempurna antara Cluster 0 dan Cluster 1. Naive Bayes sedikit lebih rendah (AUC ≈ 0.964) karena asumsi independensi fitur tidak sepenuhnya terpenuhi. Secara keseluruhan, ketiga model sangat efektif mengklasifikasikan segmen tamu.
        """)

        # Overfitting analysis
        st.markdown("#### 🔎 Analisis Overfitting & Underfitting")

        y_clf_full2 = df["cluster"].astype(int)
        X_full2 = df[FEATURES]
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
            X_full2, y_clf_full2, test_size=0.2, random_state=42, stratify=y_clf_full2
        )
        X_tr2_s = art["scaler_clf"].transform(X_tr2)
        X_te2_s = art["scaler_clf"].transform(X_te2)

        ov_rows = []
        for name, model in model_roc.items():
            tr_s = model.score(X_tr2_s, y_tr2)
            te_s = model.score(X_te2_s, y_te2)
            gap  = tr_s - te_s
            status = "✅ Fit Baik" if gap <= 0.05 else ("⚠️ Overfitting Ringan" if gap <= 0.10 else "❌ Overfitting")
            ov_rows.append({"Model": name, "Train Acc": round(tr_s,4), "Test Acc": round(te_s,4),
                            "Gap": round(gap,4), "Status": status})
        ov_df = pd.DataFrame(ov_rows)
        st.dataframe(ov_df.set_index("Model").style.background_gradient(subset=["Gap"], cmap="RdYlGn_r"),
                     use_container_width=True)

        st.markdown("""
**Interpretasi:** Semua model menunjukkan gap Train-Test yang sangat kecil (< 0.01), artinya **tidak ada indikasi overfitting maupun underfitting** yang signifikan. Model mampu menggeneralisasi dengan baik ke data baru.
        """)

    # ── TAB 4: RINGKASAN ───────────────────────────────────────────────────────
    with tab4:
        st.subheader("📋 Ringkasan Analisis")

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("### 🔵 Clustering")
            st.markdown(f"""
| Metrik | Nilai |
|--------|-------|
| Algoritma | K-Means (K=2) |
| Silhouette Score | **{meta['silhouette']}** |
| Inertia | {meta['inertia']:,.0f} |
| Cluster 0 (Budget) | {meta['cluster_counts']['0']:,} booking |
| Cluster 1 (Premium) | {meta['cluster_counts']['1']:,} booking |

**Cluster 0 — Budget / Short-stay:** ADR rendah, lead time pendek, durasi singkat.

**Cluster 1 — Premium / Long-stay:** ADR tinggi, memesan jauh hari sebelumnya, menginap lebih lama, lebih banyak tamu.
            """)

            st.markdown("### 🟡 Regresi Terbaik")
            best_r = max(meta["reg_metrics"].items(), key=lambda x: x[1]["R2"])
            st.markdown(f"""
**{best_r[0]}**
- MAE: {best_r[1]['MAE']}
- RMSE: {best_r[1]['RMSE']}
- R²: **{best_r[1]['R2']}**
            """)

        with col_r:
            st.markdown("### 🔴 Klasifikasi Terbaik")
            best_c = max(meta["clf_metrics"].items(), key=lambda x: x[1]["F1"])
            st.markdown(f"""
**{best_c[0]}**
- Accuracy: {best_c[1]['Accuracy']}
- Precision: {best_c[1]['Precision']}
- Recall: {best_c[1]['Recall']}
- F1-Score: **{best_c[1]['F1']}**
- AUC-ROC: {best_c[1]['AUC']}
            """)

            st.markdown("### 💡 Rekomendasi Bisnis")
            st.markdown("""
- **Cluster 0 (Budget):** Optimalkan revenue melalui promo last-minute, paket malam minimal, upselling add-on (sarapan, parkir).
- **Cluster 1 (Premium):** Fokus pada layanan premium — upgrade kamar, loyalty program, paket extended-stay, concierge personal.
- Gunakan fitur **Lead Time** dan **ADR** sebagai indikator utama segmentasi tamu di sistem CRM.
            """)

        st.markdown("---")
        st.markdown("### 🔑 Fitur Paling Berpengaruh")
        fi_s = pd.Series(art["rfr"].feature_importances_, index=FEATURES).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 3))
        fi_s.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(FEATURES)), edgecolor="black")
        ax.set_title("Feature Importance (Random Forest Regressor)")
        ax.set_ylabel("Importance"); ax.set_xlabel("")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.info("""
**Lead Time** adalah faktor paling dominan dalam membentuk segmen tamu, diikuti ADR dan Total Guests. 
Hal ini mengindikasikan bahwa **perilaku pemesanan** (kapan tamu memesan relatif terhadap tanggal menginap) 
lebih informatif dibanding harga kamar dalam menentukan profil tamu.
        """)
