import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Booking Demand",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "adr",
    "lead_time",
    "total_stays",
    "total_guests",
    "total_of_special_requests",
    "previous_cancellations",
]

DATA_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

# ─────────────────────────────────────────────────────────────
# PIPELINE (cached — hanya dijalankan sekali per sesi)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📥 Memuat dataset...")
def load_data():
    df = pd.read_csv(DATA_URL)
    return df


@st.cache_data(show_spinner="🧹 Membersihkan data...")
def prepare(_df):
    df = _df.copy()
    df.drop_duplicates(inplace=True)
    df["children"].fillna(0, inplace=True)
    df["country"].fillna(df["country"].mode()[0], inplace=True)
    df["agent"].fillna(0, inplace=True)
    df["company"].fillna(0, inplace=True)
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
    df["total_stays"]  = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["total_revenue"] = df["adr"] * df["total_stays"]
    df["arrival_month_num"] = df["arrival_date_month"].map(MONTH_MAP)
    for feat in ["adr", "lead_time"]:
        Q1, Q3 = df[feat].quantile(0.25), df[feat].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[feat] >= Q1 - 3*IQR) & (df[feat] <= Q3 + 3*IQR)]
    df = df[df["adr"] >= 0]
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner="🔵 K-Means Clustering...")
def run_clustering(_df):
    X = _df[FEATURES].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs, labels)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)
    return {
        "scaler": scaler, "kmeans": km, "labels": labels,
        "Xs": Xs, "Xp": Xp, "pca": pca,
        "silhouette": sil, "inertia": float(km.inertia_),
        "centers_orig": scaler.inverse_transform(km.cluster_centers_),
    }


@st.cache_resource(show_spinner="🟡 Melatih model Regresi...")
def run_regression(_df, _labels):
    X = _df[FEATURES]
    y = pd.Series(_labels, index=_df.index).astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    results, fitted, preds = [], {}, {}
    for name, m in models.items():
        m.fit(Xtr_s, ytr)
        yp = m.predict(Xte_s)
        fitted[name] = m
        preds[name]  = yp
        results.append({
            "Model": name,
            "MAE":   round(mean_absolute_error(yte, yp), 4),
            "MSE":   round(mean_squared_error(yte, yp), 4),
            "RMSE":  round(np.sqrt(mean_squared_error(yte, yp)), 4),
            "R² Score": round(r2_score(yte, yp), 4),
        })
    reg_df = pd.DataFrame(results)
    best   = reg_df.loc[reg_df["R² Score"].idxmax(), "Model"]
    return {"scaler_r": sc, "fitted": fitted, "preds": preds,
            "reg_df": reg_df, "best": best, "Xte": Xte_s, "yte": yte}


@st.cache_resource(show_spinner="🔴 Melatih model Klasifikasi...")
def run_classification(_df, _labels):
    X = _df[FEATURES]
    y = pd.Series(_labels, index=_df.index).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes":         GaussianNB(),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    results, fitted, probs = [], {}, {}
    train_scores = {}
    for name, m in models.items():
        m.fit(Xtr_s, ytr)
        yp   = m.predict(Xte_s)
        yprb = m.predict_proba(Xte_s)[:, 1]
        fitted[name] = m
        probs[name]  = yprb
        train_scores[name] = m.score(Xtr_s, ytr)
        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(yte, yp), 4),
            "Precision": round(precision_score(yte, yp), 4),
            "Recall":    round(recall_score(yte, yp), 4),
            "F1-Score":  round(f1_score(yte, yp), 4),
            "AUC-ROC":   round(roc_auc_score(yte, yprb), 4),
        })
    clf_df = pd.DataFrame(results)
    best   = clf_df.loc[clf_df["F1-Score"].idxmax(), "Model"]

    # overfitting table
    ov_rows = []
    for r in results:
        tr = train_scores[r["Model"]]
        te = r["Accuracy"]
        gap = tr - te
        if gap > 0.10:   status = "❌ Overfitting (gap > 0.10)"
        elif gap > 0.05: status = "⚠️ Overfitting Ringan (gap 0.05–0.10)"
        elif te < 0.65:  status = "⚠️ Underfitting (test score < 0.65)"
        else:            status = "✅ Fit Baik"
        ov_rows.append({"Model": r["Model"], "Acc Train": round(tr,4),
                        "Acc Test": round(te,4), "Selisih": round(gap,4), "Status": status})
    bv_df = pd.DataFrame(ov_rows)

    return {"scaler_c": sc, "fitted": fitted, "probs": probs,
            "clf_df": clf_df, "best": best,
            "Xte": Xte_s, "yte": yte, "bv_df": bv_df}


# ── Jalankan pipeline ──
with st.spinner("⚙️ Menyiapkan pipeline data & model, harap tunggu..."):
    raw_df = load_data()
    df     = prepare(raw_df)
    clust  = run_clustering(df)
    labels = clust["labels"]
    reg    = run_regression(df, labels)
    clf    = run_classification(df, labels)

centers = clust["centers_orig"].tolist()
counts_series = pd.Series(labels).value_counts().sort_index()
counts  = counts_series.to_dict()
total   = sum(counts.values())
sil     = clust["silhouette"]
inertia = clust["inertia"]

# ── Helper ──
def get_cluster_label(cl):
    # Cluster 0 = Budget / Short-stay, Cluster 1 = Premium / Long-stay
    # (berdasarkan hasil K-Means: centroid ADR Cluster 0 < Cluster 1)
    return "Budget / Short-stay" if cl == 0 else "Premium / Long-stay"


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏨 Hotel Booking\nDemand")
    st.markdown("---")
    st.markdown("**Pilih halaman:**")
    page = st.radio(
        "Navigasi",
        [
            "📊 Ringkasan Eksekutif",
            "🔵 Profil Cluster",
            "🔮 Prediksi Segmen Tamu",
            "📈 Perbandingan Model",
            "🔍 Overfitting & Underfitting",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Info Dataset**")
    st.markdown(f"- Total booking: **{total:,}**")
    st.markdown(f"- Fitur: **{len(FEATURES)}**")
    st.markdown(f"- Cluster: **2**")
    st.markdown(f"- Silhouette Score: **{sil:.4f}**")


# ─────────────────────────────────────────────────────────────
# PAGE 1 — RINGKASAN EKSEKUTIF
# ─────────────────────────────────────────────────────────────
if page == "📊 Ringkasan Eksekutif":
    st.title("📊 Ringkasan Eksekutif — Hotel Booking Demand")
    st.markdown("Dashboard analitik pipeline **K-Means Clustering → Regression → Classification**")
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Booking",    f"{total:,}")
    col2.metric("Cluster 0",        f"{counts.get(0,0):,}", f"{counts.get(0,0)/total*100:.1f}%")
    col3.metric("Cluster 1",        f"{counts.get(1,0):,}", f"{counts.get(1,0)/total*100:.1f}%")
    col4.metric("Silhouette Score", f"{sil:.4f}")
    col5.metric("Inertia",          f"{inertia:.0f}")

    st.markdown("---")

    reg_df = reg["reg_df"]
    clf_df = clf["clf_df"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("📊 RINGKASAN EKSEKUTIF — Hotel Booking Analysis", fontsize=14, fontweight="bold")

    # 1. Pie distribusi cluster
    axes[0,0].pie(
        [counts.get(0,0), counts.get(1,0)],
        labels=[f"Cluster 0\n({get_cluster_label(0)})", f"Cluster 1\n({get_cluster_label(1)})"],
        autopct="%1.1f%%", colors=["#2196F3","#FF9800"],
        startangle=90, wedgeprops={"edgecolor":"white","linewidth":2},
    )
    axes[0,0].set_title("Distribusi Cluster Tamu")

    # 2. Bar ADR
    adr_vals = [centers[0][0], centers[1][0]]
    axes[0,1].bar(["Cluster 0","Cluster 1"], adr_vals, color=["#2196F3","#FF9800"], edgecolor="black", width=0.5)
    axes[0,1].set_title("Rata-rata ADR per Cluster"); axes[0,1].set_ylabel("ADR")
    for i, v in enumerate(adr_vals): axes[0,1].text(i, v+1, f"{v:.2f}", ha="center", fontweight="bold")

    # 3. Bar Lead Time
    lt_vals = [centers[0][1], centers[1][1]]
    axes[0,2].bar(["Cluster 0","Cluster 1"], lt_vals, color=["#2196F3","#FF9800"], edgecolor="black", width=0.5)
    axes[0,2].set_title("Rata-rata Lead Time per Cluster"); axes[0,2].set_ylabel("Hari")
    for i, v in enumerate(lt_vals): axes[0,2].text(i, v+1, f"{v:.1f}", ha="center", fontweight="bold")

    # 4. Bar regresi
    x = range(len(reg_df)); w = 0.3
    axes[1,0].bar([i-w/2 for i in x], reg_df["MAE"],  w, label="MAE",  color="#ef9a9a", edgecolor="black")
    axes[1,0].bar([i+w/2 for i in x], reg_df["RMSE"], w, label="RMSE", color="#ffcc80", edgecolor="black")
    axes[1,0].set_xticks(list(x)); axes[1,0].set_xticklabels(reg_df["Model"], rotation=15, ha="right", fontsize=8)
    axes[1,0].set_title("Performa Regresi (MAE & RMSE)"); axes[1,0].legend(fontsize=8)

    # 5. Bar klasifikasi
    clf_plot = clf_df.set_index("Model")
    x2 = range(len(clf_plot)); w2 = 0.25
    metrics_bar = ["Accuracy","F1-Score","AUC-ROC"]
    colors_m = ["#a5d6a7","#80deea","#ce93d8"]
    for j, (m, col) in enumerate(zip(metrics_bar, colors_m)):
        axes[1,1].bar([i+j*w2-w2 for i in x2], clf_plot[m], w2, label=m, color=col, edgecolor="black")
    axes[1,1].set_xticks(list(x2)); axes[1,1].set_xticklabels(clf_plot.index, rotation=15, ha="right", fontsize=8)
    axes[1,1].set_title("Performa Klasifikasi"); axes[1,1].legend(fontsize=7); axes[1,1].set_ylim(0, 1.15)

    # 6. Profil cluster normalized
    centroid_df2 = pd.DataFrame(centers, columns=FEATURES, index=[0,1])
    profile_norm = (centroid_df2 - centroid_df2.min()) / (centroid_df2.max() - centroid_df2.min() + 1e-9)
    x3 = range(len(FEATURES)); w3 = 0.35
    axes[1,2].bar([i-w3/2 for i in x3], profile_norm.loc[0], w3, label="Cluster 0", color="#2196F3", alpha=0.85, edgecolor="black")
    axes[1,2].bar([i+w3/2 for i in x3], profile_norm.loc[1], w3, label="Cluster 1", color="#FF9800", alpha=0.85, edgecolor="black")
    axes[1,2].set_xticks(list(x3)); axes[1,2].set_xticklabels(FEATURES, rotation=30, ha="right", fontsize=7)
    axes[1,2].set_title("Profil Cluster (Normalized 0–1)"); axes[1,2].legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────────
# PAGE 2 — PROFIL CLUSTER
# ─────────────────────────────────────────────────────────────
elif page == "🔵 Profil Cluster":
    st.title("🔵 Profil Cluster")
    st.markdown("Karakteristik rata-rata setiap cluster hasil K-Means K=2.")
    st.markdown("---")

    col_a, col_b = st.columns(2)
    for cl, col in [(0, col_a), (1, col_b)]:
        label = get_cluster_label(cl)
        color = "#2196F3" if cl == 0 else "#FF9800"
        cnt   = counts.get(cl, 0)
        with col:
            st.markdown(f"""
            <div style='background:{color}22; border-left:5px solid {color};
                        padding:14px; border-radius:8px; margin-bottom:12px;'>
                <h3 style='color:{color}; margin:0;'>Cluster {cl}</h3>
                <h4 style='margin:4px 0;'>{label}</h4>
                <p style='margin:0;'>{cnt:,} booking ({cnt/total*100:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
            for i, feat in enumerate(FEATURES):
                st.markdown(f"- **{feat}**: `{centers[cl][i]:.2f}`")

    st.markdown("---")
    st.subheader("📍 Tabel Centroid Cluster")
    centroid_df = pd.DataFrame(
        centers, columns=FEATURES,
        index=[f"Cluster 0 ({get_cluster_label(0)})", f"Cluster 1 ({get_cluster_label(1)})"],
    ).round(2)
    st.dataframe(centroid_df.T, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Visualisasi Profil Cluster")
    centroid_df2 = pd.DataFrame(centers, columns=FEATURES, index=[0,1])
    profile_norm = (centroid_df2 - centroid_df2.min()) / (centroid_df2.max() - centroid_df2.min() + 1e-9)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for cl, color, ax in [(0, "#2196F3", axes2[0]), (1, "#FF9800", axes2[1])]:
        ax.bar(FEATURES, profile_norm.loc[cl], color=color, edgecolor="black", alpha=0.85)
        ax.set_title(f"Profil Cluster {cl} — {get_cluster_label(cl)} (Normalized 0–1)")
        ax.set_ylim(0, 1.15); ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(profile_norm.loc[cl]):
            ax.text(i, v+0.02, f"{centers[cl][i]:.1f}", ha="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2); plt.close()


# ─────────────────────────────────────────────────────────────
# PAGE 3 — PREDIKSI  ← FIXED: scaler konsisten
# ─────────────────────────────────────────────────────────────
elif page == "🔮 Prediksi Segmen Tamu":
    st.title("🔮 Prediksi Segmen Tamu")
    st.markdown("Masukkan parameter tamu baru untuk memprediksi cluster segmennya.")
    st.markdown("---")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Input Parameter Tamu")
        adr         = st.slider("ADR (harga/malam)",        0.0, 500.0, 100.0, 5.0)
        lead_time   = st.slider("Lead Time (hari)",          0,   400,   30,    5)
        total_stays = st.slider("Total Stays (malam)",       0,   20,    3,     1)
        total_guests= st.slider("Total Guests",              1,   10,    2,     1)
        special_req = st.slider("Special Requests",          0,   5,     0,     1)
        prev_cancel = st.slider("Previous Cancellations",    0,   10,    0,     1)
        predict_btn = st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Hasil Prediksi")

        if predict_btn:
            # ── Input array ──
            input_arr = np.array([[adr, lead_time, total_stays, total_guests, special_req, prev_cancel]])
            input_df  = pd.DataFrame(input_arr, columns=FEATURES)

            # ── Tiap model punya scaler-nya sendiri — gunakan scaler_c untuk clf ──
            input_scaled_c = clf["scaler_c"].transform(input_df)

            votes, rows = [], []
            for name, model in clf["fitted"].items():
                pred  = int(model.predict(input_scaled_c)[0])
                prob  = model.predict_proba(input_scaled_c)[0]
                votes.append(pred)
                label = "Premium / Long-stay" if pred == 1 else "Budget / Short-stay"
                rows.append({
                    "Model":        name,
                    "Prediksi":     f"Cluster {pred} ({label})",
                    "P(Cluster 0)": f"{prob[0]:.1%}",
                    "P(Cluster 1)": f"{prob[1]:.1%}",
                })

            final       = max(set(votes), key=votes.count)
            final_label = "Premium / Long-stay" if final == 1 else "Budget / Short-stay"

            st.success(f"🏆 Prediksi Final (majority vote): **Cluster {final} — {final_label}**")
            st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

            # ── Probability bar chart — gunakan model terbaik ──
            best_clf_name = clf["best"]
            probs_best = clf["fitted"][best_clf_name].predict_proba(input_scaled_c)[0]
            fig3, ax3 = plt.subplots(figsize=(7, 2.5))
            bars3 = ax3.barh(
                ["Cluster 0 (Budget / Short-stay)", "Cluster 1 (Premium / Long-stay)"],
                probs_best,
                color=["#2196F3","#FF9800"], edgecolor="black",
            )
            ax3.set_xlim(0, 1)
            ax3.set_title(f"Probabilitas Prediksi ({best_clf_name})")
            ax3.set_xlabel("Probabilitas")
            for bar, p in zip(bars3, probs_best):
                ax3.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                         f"{p:.1%}", va="center", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig3); plt.close()

            # ── Regresi juga ──
            input_scaled_r = reg["scaler_r"].transform(input_df)
            reg_val = reg["fitted"][reg["best"]].predict(input_scaled_r)[0]
            st.info(f"📈 Prediksi Regresi ({reg['best']}): `{reg_val:.4f}` → lebih dekat ke Cluster **{'0' if reg_val < 0.5 else '1'}**")

        else:
            st.info("👈 Atur input di sebelah kiri lalu klik **Prediksi Sekarang**")
            st.markdown("**Referensi Centroid:**")
            ref_df = pd.DataFrame({
                "Fitur":                            FEATURES,
                f"Cluster 0 ({get_cluster_label(0)})": [round(c, 2) for c in centers[0]],
                f"Cluster 1 ({get_cluster_label(1)})": [round(c, 2) for c in centers[1]],
            }).set_index("Fitur")
            st.dataframe(ref_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 4 — PERBANDINGAN MODEL
# ─────────────────────────────────────────────────────────────
elif page == "📈 Perbandingan Model":
    st.title("📈 Perbandingan Performa Model")
    st.markdown("---")

    reg_df = reg["reg_df"]
    clf_df = clf["clf_df"]

    tab_reg, tab_clf = st.tabs(["📈 Regresi", "🎯 Klasifikasi"])

    with tab_reg:
        st.subheader("Model Regresi")
        best_reg = reg["best"]
        st.markdown(f"**Model terbaik:** `{best_reg}`")
        st.dataframe(reg_df.set_index("Model"), use_container_width=True)

        fig_r, axes_r = plt.subplots(1, 2, figsize=(14, 5))
        x = range(len(reg_df)); w = 0.2
        axes_r[0].bar([i-w for i in x],  reg_df["MAE"],      w, label="MAE",      color="#ef9a9a", edgecolor="black")
        axes_r[0].bar(list(x),            reg_df["RMSE"],     w, label="RMSE",     color="#ffcc80", edgecolor="black")
        axes_r[0].bar([i+w for i in x],  reg_df["R² Score"], w, label="R² Score", color="#a5d6a7", edgecolor="black")
        axes_r[0].set_xticks(list(x)); axes_r[0].set_xticklabels(reg_df["Model"], rotation=15, ha="right")
        axes_r[0].set_title("Perbandingan Metrik Regresi"); axes_r[0].legend(); axes_r[0].set_ylim(0, 1.1)

        axes_r[1].bar(reg_df["Model"], reg_df["MSE"], color="#b39ddb", edgecolor="black")
        axes_r[1].set_title("MSE per Model"); axes_r[1].set_ylabel("MSE")
        axes_r[1].tick_params(axis="x", rotation=15)
        for i, v in enumerate(reg_df["MSE"]):
            axes_r[1].text(i, v+0.001, f"{v:.4f}", ha="center", fontsize=9)
        plt.tight_layout(); st.pyplot(fig_r); plt.close()

        r2_val = float(reg_df[reg_df["Model"]==best_reg]["R² Score"].values[0])
        kualitas = "🟢 Sangat Baik" if r2_val >= 0.9 else ("🟡 Baik" if r2_val >= 0.7 else "🟠 Cukup")
        st.info(f"**Interpretasi:** {best_reg} menghasilkan performa terbaik. Kualitas R²: {kualitas}.")

    with tab_clf:
        st.subheader("Model Klasifikasi")
        best_clf = clf["best"]
        st.markdown(f"**Model terbaik:** `{best_clf}`")
        st.dataframe(clf_df.set_index("Model"), use_container_width=True)

        fig_c, ax_c = plt.subplots(figsize=(12, 5))
        metrics  = ["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]
        colors_m = ["#80cbc4","#80deea","#b39ddb","#ef9a9a","#ffcc80"]
        x2 = range(len(clf_df)); w2 = 0.15
        for j, (m, col) in enumerate(zip(metrics, colors_m)):
            ax_c.bar([i+j*w2-2*w2 for i in x2], clf_df[m], w2, label=m, color=col, edgecolor="black")
        ax_c.set_xticks(list(x2)); ax_c.set_xticklabels(clf_df["Model"], rotation=15, ha="right")
        ax_c.set_title("Perbandingan Semua Metrik Klasifikasi"); ax_c.legend(fontsize=8); ax_c.set_ylim(0, 1.15)
        plt.tight_layout(); st.pyplot(fig_c); plt.close()

        f1_val = float(clf_df[clf_df["Model"]==best_clf]["F1-Score"].values[0])
        kualitas_c = "🟢 Sangat Baik" if f1_val >= 0.9 else ("🟡 Baik" if f1_val >= 0.75 else "🟠 Cukup")
        st.info(f"**Interpretasi:** {best_clf} dipilih karena F1-Score tertinggi ({f1_val}). Kualitas: {kualitas_c}.")


# ─────────────────────────────────────────────────────────────
# PAGE 5 — OVERFITTING & UNDERFITTING
# ─────────────────────────────────────────────────────────────
elif page == "🔍 Overfitting & Underfitting":
    st.title("🔍 Analisis Overfitting & Underfitting")
    st.markdown("---")

    st.subheader("📖 Konsep")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**⚠️ Underfitting**")
    c1.markdown("Model terlalu sederhana. Bahkan data training pun tidak bisa diprediksi baik.")
    c1.markdown("**Ciri:** Akurasi training rendah (<65%)")
    c2.markdown("**✅ Fit Baik**")
    c2.markdown("Model belajar pola dengan benar, tidak hafal, tidak terlalu sederhana.")
    c2.markdown("**Ciri:** Gap train–test kecil (<5%)")
    c3.markdown("**⚠️ Overfitting**")
    c3.markdown("Model hafal data training, tidak bisa generalisasi ke data baru.")
    c3.markdown("**Ciri:** Gap train–test besar (>10%)")

    st.markdown("---")
    bv_df = clf["bv_df"]

    st.subheader("📊 Train vs Test Accuracy — Model Klasifikasi")
    fig_ov, ax_ov = plt.subplots(figsize=(10, 4))
    x = np.arange(len(bv_df)); w = 0.35
    ax_ov.bar(x-w/2, bv_df["Acc Train"], w, label="Train Accuracy", color="#2196F3", alpha=0.85, edgecolor="black")
    ax_ov.bar(x+w/2, bv_df["Acc Test"],  w, label="Test Accuracy",  color="#FF9800", alpha=0.85, edgecolor="black")
    ax_ov.set_xticks(x); ax_ov.set_xticklabels(bv_df["Model"], rotation=10, ha="right")
    ax_ov.set_ylabel("Accuracy"); ax_ov.set_ylim(0, 1.15)
    ax_ov.set_title("Train vs Test Accuracy per Model"); ax_ov.legend()
    ax_ov.axhline(0.65, color="red", linestyle=":", linewidth=1, alpha=0.5)
    plt.tight_layout(); st.pyplot(fig_ov); plt.close()

    st.subheader("📉 Gap (Train − Test) per Model")
    fig_gap, ax_gap = plt.subplots(figsize=(8, 4))
    colors_gap = ["#ffc7ce" if g>0.10 else ("#ffeb9c" if g>0.05 else "#c6efce") for g in bv_df["Selisih"]]
    bars_gap = ax_gap.bar(bv_df["Model"], bv_df["Selisih"], color=colors_gap, edgecolor="black")
    ax_gap.axhline(0.05, color="orange", linestyle="--", linewidth=1.5, label="Threshold ringan (0.05)")
    ax_gap.axhline(0.10, color="red",    linestyle="--", linewidth=1.5, label="Threshold overfitting (0.10)")
    for bar, gap in zip(bars_gap, bv_df["Selisih"]):
        ax_gap.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f"{gap:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax_gap.set_title("Gap Train–Test per Model\n(Makin kecil = makin baik)")
    ax_gap.set_ylabel("Gap Score"); ax_gap.legend(fontsize=9)
    ax_gap.set_ylim(0, max(bv_df["Selisih"].max()*1.5, 0.15))
    plt.xticks(rotation=10, ha="right"); plt.tight_layout(); st.pyplot(fig_gap); plt.close()

    st.subheader("📋 Tabel Status Model")
    st.dataframe(bv_df.set_index("Model"), use_container_width=True)

    st.markdown("---")
    st.subheader("✍️ Kesimpulan")
    ok  = sum(1 for s in bv_df["Status"] if "✅" in s)
    ov  = sum(1 for s in bv_df["Status"] if "❌" in s or ("⚠️" in s and "Over" in s))
    und = sum(1 for s in bv_df["Status"] if "⚠️" in s and "Under" in s)
    m1, m2, m3 = st.columns(3)
    m1.metric("✅ Fit Baik",    ok)
    m2.metric("⚠️ Overfitting",  ov)
    m3.metric("⚠️ Underfitting", und)
    if ov == 0 and und == 0:
        st.success("✅ Semua model dalam kondisi Fit Baik — tidak ada overfitting maupun underfitting yang signifikan.")
    else:
        for _, row in bv_df.iterrows():
            st.markdown(f"- **{row['Model']}**: {row['Status']}")
