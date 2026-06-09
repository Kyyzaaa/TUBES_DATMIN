import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Booking Demand",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    base = "models"
    km          = joblib.load(f"{base}/kmeans_model.pkl")
    sc          = joblib.load(f"{base}/scaler.pkl")
    sc_r        = joblib.load(f"{base}/scaler_r.pkl")
    sc_c        = joblib.load(f"{base}/scaler_c.pkl")
    meta        = joblib.load(f"{base}/model_metadata.pkl")
    reg_models  = joblib.load(f"{base}/reg_models.pkl")
    clf_models  = joblib.load(f"{base}/clf_models.pkl")
    reg_df      = joblib.load(f"{base}/reg_df.pkl")
    clf_df      = joblib.load(f"{base}/clf_df.pkl")
    overfit     = joblib.load(f"{base}/overfit_meta.pkl")
    # pastikan clf_models adalah dict
    if not isinstance(clf_models, dict):
        clf_models = {"Model": clf_models}
    return km, sc, sc_r, sc_c, meta, reg_models, clf_models, reg_df, clf_df, overfit

try:
    kmeans, scaler, scaler_r, scaler_c, meta, reg_models, clf_models, reg_df, clf_df, overfit_meta = load_all()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

FEATURES = [
    'adr',
    'lead_time',
    'total_stays',
    'total_guests',
    'total_of_special_requests',
    'previous_cancellations'
]

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏨 Hotel Booking\nDemand")
    st.markdown("---")
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
        label_visibility="collapsed"
    )

    st.markdown("---")
    if models_loaded:
        counts = meta['cluster_counts']
        total  = sum(counts.values())
        st.markdown("**Info Dataset**")
        st.markdown(f"- Total booking: **{total:,}**")
        st.markdown(f"- Fitur: **{len(FEATURES)}**")
        st.markdown(f"- Cluster: **2**")
        st.markdown(f"- Silhouette Score: **{meta['silhouette_score']:.4f}**")

# ─────────────────────────────────────────────────────────────
# CEK MODEL
# ─────────────────────────────────────────────────────────────
if not models_loaded:
    st.error(f"❌ Folder `models/` tidak ditemukan atau tidak lengkap.")
    st.code(f"Error: {load_error}")
    st.info("""
    **Cara fix:**
    1. Jalankan notebook Colab sampai selesai (termasuk cell deployment)
    2. Download folder `models/` dari Colab
    3. Taruh folder `models/` di repo GitHub bersama `app.py`
    """)
    st.stop()

# Helper
counts  = meta['cluster_counts']
total   = sum(counts.values())
centers = meta['cluster_centers']

def get_cluster_label(cl):
    c0_adr = centers[0][0]
    c1_adr = centers[1][0]
    if cl == 0:
        return "Premium / Long-stay" if c0_adr > c1_adr else "Budget / Short-stay"
    else:
        return "Premium / Long-stay" if c1_adr > c0_adr else "Budget / Short-stay"

# ─────────────────────────────────────────────────────────────
# PAGE 1 — RINGKASAN EKSEKUTIF
# ─────────────────────────────────────────────────────────────
if page == "📊 Ringkasan Eksekutif":
    st.title("📊 Ringkasan Eksekutif — Hotel Booking Demand")
    st.markdown("Dashboard analitik pipeline **K-Means Clustering → Regression → Classification**")
    st.markdown("---")

    # Metric row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Booking",       f"{total:,}")
    col2.metric("Cluster 0",           f"{counts.get(0,0):,}", f"{counts.get(0,0)/total*100:.1f}%")
    col3.metric("Cluster 1",           f"{counts.get(1,0):,}", f"{counts.get(1,0)/total*100:.1f}%")
    col4.metric("Silhouette Score",    f"{meta['silhouette_score']:.4f}")
    col5.metric("Inertia",             f"{meta['inertia']:.0f}")

    st.markdown("---")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('📊 RINGKASAN EKSEKUTIF — Hotel Booking Analysis', fontsize=14, fontweight='bold')

    # 1. Pie distribusi cluster
    axes[0,0].pie(
        [counts.get(0,0), counts.get(1,0)],
        labels=[f'Cluster 0\n({get_cluster_label(0)})', f'Cluster 1\n({get_cluster_label(1)})'],
        autopct='%1.1f%%', colors=['#2196F3','#FF9800'],
        startangle=90, wedgeprops={'edgecolor':'white','linewidth':2}
    )
    axes[0,0].set_title('Distribusi Cluster Tamu')

    # 2. Bar centroid ADR
    adr_vals = [centers[0][0], centers[1][0]]
    axes[0,1].bar(['Cluster 0', 'Cluster 1'], adr_vals,
                   color=['#2196F3','#FF9800'], edgecolor='black', width=0.5)
    axes[0,1].set_title('Rata-rata ADR per Cluster')
    axes[0,1].set_ylabel('ADR')
    for i, v in enumerate(adr_vals):
        axes[0,1].text(i, v+1, f'{v:.2f}', ha='center', fontweight='bold')

    # 3. Bar centroid lead_time
    lt_vals = [centers[0][1], centers[1][1]]
    axes[0,2].bar(['Cluster 0', 'Cluster 1'], lt_vals,
                   color=['#2196F3','#FF9800'], edgecolor='black', width=0.5)
    axes[0,2].set_title('Rata-rata Lead Time per Cluster')
    axes[0,2].set_ylabel('Hari')
    for i, v in enumerate(lt_vals):
        axes[0,2].text(i, v+1, f'{v:.1f}', ha='center', fontweight='bold')

    # 4. Bar regresi MAE & RMSE
    reg_plot = reg_df.set_index('Model')
    x = range(len(reg_plot))
    w = 0.3
    axes[1,0].bar([i-w/2 for i in x], reg_plot['MAE'],  w, label='MAE',  color='#ef9a9a', edgecolor='black')
    axes[1,0].bar([i+w/2 for i in x], reg_plot['RMSE'], w, label='RMSE', color='#ffcc80', edgecolor='black')
    axes[1,0].set_xticks(list(x))
    axes[1,0].set_xticklabels(reg_plot.index, rotation=15, ha='right', fontsize=8)
    axes[1,0].set_title('Performa Regresi (MAE & RMSE)')
    axes[1,0].legend(fontsize=8)

    # 5. Bar klasifikasi
    clf_plot = clf_df.set_index('Model')
    x2 = range(len(clf_plot))
    metrics = ['Accuracy','F1-Score','AUC-ROC']
    colors_m = ['#a5d6a7','#80deea','#ce93d8']
    w2 = 0.25
    for j, (m, col) in enumerate(zip(metrics, colors_m)):
        axes[1,1].bar([i + j*w2 - w2 for i in x2], clf_plot[m], w2,
                       label=m, color=col, edgecolor='black')
    axes[1,1].set_xticks(list(x2))
    axes[1,1].set_xticklabels(clf_plot.index, rotation=15, ha='right', fontsize=8)
    axes[1,1].set_title('Performa Klasifikasi')
    axes[1,1].legend(fontsize=7)
    axes[1,1].set_ylim(0, 1.15)

    # 6. Profil cluster normalized
    centroid_df = pd.DataFrame(centers, columns=FEATURES, index=[0, 1])
    profile_norm = (centroid_df - centroid_df.min()) / (centroid_df.max() - centroid_df.min() + 1e-9)
    x3 = range(len(FEATURES))
    w3 = 0.35
    axes[1,2].bar([i-w3/2 for i in x3], profile_norm.loc[0], w3, label='Cluster 0', color='#2196F3', alpha=0.85, edgecolor='black')
    axes[1,2].bar([i+w3/2 for i in x3], profile_norm.loc[1], w3, label='Cluster 1', color='#FF9800', alpha=0.85, edgecolor='black')
    axes[1,2].set_xticks(list(x3))
    axes[1,2].set_xticklabels(FEATURES, rotation=30, ha='right', fontsize=7)
    axes[1,2].set_title('Profil Cluster (Normalized 0–1)')
    axes[1,2].legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

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
                val = centers[cl][i]
                st.markdown(f"- **{feat}**: `{val:.2f}`")

    st.markdown("---")
    st.subheader("📍 Tabel Centroid Cluster")
    centroid_df = pd.DataFrame(
        centers, columns=FEATURES,
        index=[f"Cluster 0 ({get_cluster_label(0)})", f"Cluster 1 ({get_cluster_label(1)})"]
    ).round(2)
    st.dataframe(centroid_df.T, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Visualisasi Profil Cluster")
    centroid_df2 = pd.DataFrame(centers, columns=FEATURES, index=[0, 1])
    profile_norm = (centroid_df2 - centroid_df2.min()) / (centroid_df2.max() - centroid_df2.min() + 1e-9)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for cl, color, ax in [(0, '#2196F3', axes2[0]), (1, '#FF9800', axes2[1])]:
        ax.bar(FEATURES, profile_norm.loc[cl], color=color, edgecolor='black', alpha=0.85)
        ax.set_title(f'Profil Cluster {cl} — {get_cluster_label(cl)} (Normalized 0–1)')
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis='x', rotation=30)
        for i, v in enumerate(profile_norm.loc[cl]):
            ax.text(i, v + 0.02, f'{centers[cl][i]:.1f}', ha='center', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ─────────────────────────────────────────────────────────────
# PAGE 3 — PREDIKSI
# ─────────────────────────────────────────────────────────────
elif page == "🔮 Prediksi Segmen Tamu":
    st.title("🔮 Prediksi Segmen Tamu")
    st.markdown("Masukkan parameter tamu baru untuk memprediksi cluster segmennya.")
    st.markdown("---")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Input Parameter Tamu")
        adr         = st.slider("ADR (harga/malam)",          0.0, 500.0, 100.0, 5.0)
        lead_time   = st.slider("Lead Time (hari)",            0,   400,   30,    5)
        total_stays = st.slider("Total Stays (malam)",         0,   20,    3,     1)
        total_guests= st.slider("Total Guests",                1,   10,    2,     1)
        special_req = st.slider("Special Requests",            0,   5,     0,     1)
        prev_cancel = st.slider("Previous Cancellations",      0,   10,    0,     1)

        predict_btn = st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Hasil Prediksi")

        if predict_btn:
            input_arr    = np.array([[adr, lead_time, total_stays, total_guests, special_req, prev_cancel]])
            input_scaled = scaler_c.transform(input_arr)

            votes = []
            rows  = []
            for name, model in clf_models.items():
                pred  = model.predict(input_scaled)[0]
                prob  = model.predict_proba(input_scaled)[0]
                votes.append(pred)
                label = get_cluster_label(pred)
                rows.append({
                    'Model'       : name,
                    'Prediksi'    : f'Cluster {pred} ({label})',
                    'P(Cluster 0)': f'{prob[0]:.1%}',
                    'P(Cluster 1)': f'{prob[1]:.1%}'
                })

            final       = max(set(votes), key=votes.count)
            final_label = get_cluster_label(final)
            final_color = "#2196F3" if final == 0 else "#FF9800"

            st.markdown(f"""
            <div style='background:{final_color}22; border-left:5px solid {final_color};
                        padding:16px; border-radius:8px; margin-bottom:16px;'>
                <h3 style='color:{final_color}; margin:0;'>
                    🏆 Prediksi Final (majority vote): Cluster {final} — {final_label}
                </h3>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

            # Bar chart probabilitas (Logistic Regression)
            lr_model = clf_models.get('Logistic Regression')
            if lr_model:
                probs_lr = lr_model.predict_proba(input_scaled)[0]
                fig3, ax3 = plt.subplots(figsize=(7, 2.5))
                bars3 = ax3.barh(
                    [f'Cluster 0 ({get_cluster_label(0)})',
                     f'Cluster 1 ({get_cluster_label(1)})'],
                    probs_lr,
                    color=['#2196F3','#FF9800'], edgecolor='black'
                )
                ax3.set_xlim(0, 1)
                ax3.set_title('Probabilitas Prediksi (Logistic Regression)')
                ax3.set_xlabel('Probabilitas')
                for bar, p in zip(bars3, probs_lr):
                    ax3.text(bar.get_width() + 0.01,
                             bar.get_y() + bar.get_height()/2,
                             f'{p:.1%}', va='center', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()

        else:
            st.info("👈 Atur input di sebelah kiri lalu klik **Prediksi Sekarang**")
            st.markdown("**Referensi Centroid:**")
            ref_df = pd.DataFrame({
                "Fitur"                        : FEATURES,
                f"Cluster 0 ({get_cluster_label(0)})": [round(c, 2) for c in centers[0]],
                f"Cluster 1 ({get_cluster_label(1)})": [round(c, 2) for c in centers[1]]
            }).set_index("Fitur")
            st.dataframe(ref_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 4 — PERBANDINGAN MODEL
# ─────────────────────────────────────────────────────────────
elif page == "📈 Perbandingan Model":
    st.title("📈 Perbandingan Performa Model")
    st.markdown("---")

    tab_reg, tab_clf = st.tabs(["📈 Regresi", "🎯 Klasifikasi"])

    with tab_reg:
        st.subheader("Model Regresi")
        st.markdown(f"**Target:** Label cluster (0/1) hasil K-Means")
        best_reg = reg_df.loc[reg_df['R² Score'].idxmax(), 'Model']
        st.markdown(f"**Model terbaik:** `{best_reg}`")
        st.dataframe(
            reg_df.set_index('Model'),
            use_container_width=True
        )

        fig_r, axes_r = plt.subplots(1, 2, figsize=(14, 5))
        x = range(len(reg_df))
        w = 0.2
        axes_r[0].bar([i-w for i in x], reg_df['MAE'],      w, label='MAE',      color='#ef9a9a', edgecolor='black')
        axes_r[0].bar(list(x),           reg_df['RMSE'],     w, label='RMSE',     color='#ffcc80', edgecolor='black')
        axes_r[0].bar([i+w for i in x], reg_df['R² Score'], w, label='R² Score', color='#a5d6a7', edgecolor='black')
        axes_r[0].set_xticks(list(x))
        axes_r[0].set_xticklabels(reg_df['Model'], rotation=15, ha='right')
        axes_r[0].set_title('Perbandingan Metrik Regresi')
        axes_r[0].legend()
        axes_r[0].set_ylim(0, 1.1)

        axes_r[1].bar(reg_df['Model'], reg_df['MSE'], color='#b39ddb', edgecolor='black')
        axes_r[1].set_title('MSE per Model')
        axes_r[1].set_ylabel('MSE')
        axes_r[1].tick_params(axis='x', rotation=15)
        for i, v in enumerate(reg_df['MSE']):
            axes_r[1].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig_r)
        plt.close()

        r2_val = float(reg_df[reg_df['Model']==best_reg]['R² Score'].values[0])
        if r2_val >= 0.9:   kualitas = "🟢 Sangat Baik"
        elif r2_val >= 0.7: kualitas = "🟡 Baik"
        elif r2_val >= 0.5: kualitas = "🟠 Cukup"
        else:               kualitas = "🔴 Rendah (normal untuk binary target)"
        st.info(f"**Interpretasi:** {best_reg} menghasilkan performa terbaik. Kualitas R²: {kualitas}. Target berupa binary cluster label (0/1) sehingga R² yang lebih rendah adalah normal.")

    with tab_clf:
        st.subheader("Model Klasifikasi")
        st.markdown(f"**Target:** Label cluster (0/1) hasil K-Means")
        best_clf = clf_df.loc[clf_df['F1-Score'].idxmax(), 'Model']
        st.markdown(f"**Model terbaik:** `{best_clf}`")
        st.dataframe(
            clf_df.set_index('Model'),
            use_container_width=True
        )

        fig_c, ax_c = plt.subplots(figsize=(12, 5))
        metrics  = ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']
        colors_m = ['#80cbc4','#80deea','#b39ddb','#ef9a9a','#ffcc80']
        x2 = range(len(clf_df))
        w2 = 0.15
        for j, (m, col) in enumerate(zip(metrics, colors_m)):
            ax_c.bar([i + j*w2 - 2*w2 for i in x2], clf_df[m], w2,
                      label=m, color=col, edgecolor='black')
        ax_c.set_xticks(list(x2))
        ax_c.set_xticklabels(clf_df['Model'], rotation=15, ha='right')
        ax_c.set_title('Perbandingan Semua Metrik Klasifikasi')
        ax_c.legend(fontsize=8)
        ax_c.set_ylim(0, 1.15)
        plt.tight_layout()
        st.pyplot(fig_c)
        plt.close()

        f1_val = float(clf_df[clf_df['Model']==best_clf]['F1-Score'].values[0])
        if f1_val >= 0.9:    kualitas_c = "🟢 Sangat Baik"
        elif f1_val >= 0.75: kualitas_c = "🟡 Baik"
        else:                kualitas_c = "🟠 Cukup"
        st.info(f"**Interpretasi:** {best_clf} dipilih karena F1-Score tertinggi ({f1_val}). Kualitas: {kualitas_c}. F1-Score adalah metrik paling robust untuk data dengan distribusi kelas tidak seimbang.")

# ─────────────────────────────────────────────────────────────
# PAGE 5 — OVERFITTING & UNDERFITTING
# ─────────────────────────────────────────────────────────────
elif page == "🔍 Overfitting & Underfitting":
    st.title("🔍 Analisis Overfitting & Underfitting")
    st.markdown("---")

    # Penjelasan konsep
    st.subheader("📖 Konsep")
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    <div style='background:#fff3cd;border-left:5px solid #ffc107;padding:14px;border-radius:8px;min-height:140px'>
    <h4>⚠️ Underfitting</h4>
    <p>Model terlalu sederhana. Bahkan data training pun tidak bisa diprediksi baik.</p>
    <b>Ciri:</b> Akurasi training rendah (&lt;65%)
    </div>
    """, unsafe_allow_html=True)
    c2.markdown("""
    <div style='background:#d4edda;border-left:5px solid #28a745;padding:14px;border-radius:8px;min-height:140px'>
    <h4>✅ Fit Baik</h4>
    <p>Model belajar pola dengan benar, tidak hafal, tidak terlalu sederhana.</p>
    <b>Ciri:</b> Gap train–test kecil (&lt;5%)
    </div>
    """, unsafe_allow_html=True)
    c3.markdown("""
    <div style='background:#f8d7da;border-left:5px solid #dc3545;padding:14px;border-radius:8px;min-height:140px'>
    <h4>⚠️ Overfitting</h4>
    <p>Model hafal data training, tidak bisa generalisasi ke data baru.</p>
    <b>Ciri:</b> Gap train–test besar (&gt;10%)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if overfit_meta is None:
        st.warning("⚠️ Data belum tersedia. Pastikan `overfit_meta.pkl` ada di folder `models/`.")
    else:
        bv_df = pd.DataFrame(overfit_meta['clf_fit'])

        # ── Train vs Test Bar Chart ──
        st.subheader("📊 Train vs Test Accuracy — Model Klasifikasi")
        fig_ov, ax_ov = plt.subplots(figsize=(10, 4))
        x = np.arange(len(bv_df))
        w = 0.35
        ax_ov.bar(x - w/2, bv_df['Acc Train'], w, label='Train Accuracy', color='#2196F3', alpha=0.85, edgecolor='black')
        ax_ov.bar(x + w/2, bv_df['Acc Test'],  w, label='Test Accuracy',  color='#FF9800', alpha=0.85, edgecolor='black')
        ax_ov.set_xticks(x)
        ax_ov.set_xticklabels(bv_df['Model'], rotation=10, ha='right')
        ax_ov.set_ylabel('Accuracy')
        ax_ov.set_ylim(0, 1.15)
        ax_ov.set_title('Train vs Test Accuracy per Model')
        ax_ov.legend()
        ax_ov.axhline(0.65, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Min threshold (0.65)')
        plt.tight_layout()
        st.pyplot(fig_ov)
        plt.close()

        # ── Gap Chart ──
        st.subheader("📉 Gap (Train − Test) per Model")
        fig_gap, ax_gap = plt.subplots(figsize=(8, 4))
        colors_gap = []
        for g in bv_df['Selisih']:
            if g > 0.10:   colors_gap.append('#ffc7ce')
            elif g > 0.05: colors_gap.append('#ffeb9c')
            else:          colors_gap.append('#c6efce')
        bars_gap = ax_gap.bar(bv_df['Model'], bv_df['Selisih'],
                               color=colors_gap, edgecolor='black')
        ax_gap.axhline(0.05, color='orange', linestyle='--', linewidth=1.5, label='Threshold ringan (0.05)')
        ax_gap.axhline(0.10, color='red',    linestyle='--', linewidth=1.5, label='Threshold overfitting (0.10)')
        for bar, gap in zip(bars_gap, bv_df['Selisih']):
            ax_gap.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                        f'{gap:.4f}', ha='center', fontsize=9, fontweight='bold')
        ax_gap.set_title('Gap Train–Test per Model\n(Makin kecil = makin baik)')
        ax_gap.set_ylabel('Gap Score')
        ax_gap.legend(fontsize=9)
        ax_gap.set_ylim(0, max(bv_df['Selisih'].max()*1.5, 0.15))
        plt.xticks(rotation=10, ha='right')
        plt.tight_layout()
        st.pyplot(fig_gap)
        plt.close()

        # ── Tabel status ──
        st.subheader("📋 Tabel Status Model")
        st.dataframe(
            bv_df.set_index('Model'),
            use_container_width=True
        )

        # ── Kesimpulan ──
        st.markdown("---")
        st.subheader("✍️ Kesimpulan")
        ok  = sum(1 for s in bv_df['Status'] if '✅' in s)
        ov  = sum(1 for s in bv_df['Status'] if '❌' in s or ('⚠️' in s and 'Over' in s))
        und = sum(1 for s in bv_df['Status'] if '⚠️' in s and 'Under' in s)

        m1, m2, m3 = st.columns(3)
        m1.metric("✅ Fit Baik",    ok)
        m2.metric("⚠️ Overfitting",  ov)
        m3.metric("⚠️ Underfitting", und)

        if ov == 0 and und == 0:
            st.success("✅ Semua model dalam kondisi Fit Baik — tidak ada overfitting maupun underfitting yang signifikan.")
        else:
            for _, row in bv_df.iterrows():
                st.markdown(f"- **{row['Model']}**: {row['Status']}")
