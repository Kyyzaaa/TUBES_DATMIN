import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

st.set_page_config(
    page_title="🏨 Hotel Booking Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(show_spinner="Memuat dataset dan melatih model...")
def load_and_train():
    url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv'
    df = pd.read_csv(url)

    df_clean = df.copy()
    df_clean.drop_duplicates(inplace=True)
    df_clean['children'].fillna(0, inplace=True)
    df_clean['country'].fillna(df_clean['country'].mode()[0], inplace=True)
    df_clean['agent'].fillna(0, inplace=True)
    df_clean['company'].fillna(0, inplace=True)
    df_clean = df_clean[(df_clean['adr'] >= 0) & (df_clean['adr'] <= 5000)]
    df_clean = df_clean[(df_clean['adults'] + df_clean['children'] + df_clean['babies']) > 0]
    df_clean['total_stays']   = df_clean['stays_in_weekend_nights'] + df_clean['stays_in_week_nights']
    df_clean['total_guests']  = df_clean['adults'] + df_clean['children'] + df_clean['babies']
    df_clean['total_revenue'] = df_clean['adr'] * df_clean['total_stays']

    selected_features = [
        'adr', 'lead_time', 'total_stays', 'total_guests',
        'total_of_special_requests', 'previous_cancellations'
    ]

    X = df_clean[selected_features].copy()
    scaler_km = StandardScaler()
    X_scaled  = scaler_km.fit_transform(X)
    kmeans    = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
    df_clean['cluster'] = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, df_clean['cluster'])

    y_reg = df_clean['cluster'].astype(float)
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    scaler_r = StandardScaler()
    X_tr_rs  = scaler_r.fit_transform(X_tr_r)
    X_te_rs  = scaler_r.transform(X_te_r)

    reg_models = {
        'Linear Regression'       : LinearRegression(),
        'Random Forest Regressor' : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    reg_results = []
    for name, m in reg_models.items():
        m.fit(X_tr_rs, y_tr_r)
        yp = m.predict(X_te_rs)
        reg_results.append({
            'Model'   : name,
            'MAE'     : round(mean_absolute_error(y_te_r, yp), 4),
            'MSE'     : round(mean_squared_error(y_te_r, yp), 4),
            'RMSE'    : round(np.sqrt(mean_squared_error(y_te_r, yp)), 4),
            'R² Score': round(r2_score(y_te_r, yp), 4)
        })
    reg_df = pd.DataFrame(reg_results)

    y_clf = df_clean['cluster'].astype(int)
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    scaler_c = StandardScaler()
    X_tr_cs  = scaler_c.fit_transform(X_tr_c)
    X_te_cs  = scaler_c.transform(X_te_c)

    clf_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes'        : GaussianNB(),
        'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    clf_results = []
    clf_fitted  = {}
    for name, m in clf_models.items():
        m.fit(X_tr_cs, y_tr_c)
        yp    = m.predict(X_te_cs)
        yprob = m.predict_proba(X_te_cs)[:, 1]
        clf_results.append({
            'Model'    : name,
            'Accuracy' : round(accuracy_score(y_te_c, yp), 4),
            'Precision': round(precision_score(y_te_c, yp), 4),
            'Recall'   : round(recall_score(y_te_c, yp), 4),
            'F1-Score' : round(f1_score(y_te_c, yp), 4),
            'AUC-ROC'  : round(roc_auc_score(y_te_c, yprob), 4)
        })
        clf_fitted[name] = m
    clf_df = pd.DataFrame(clf_results)

    bv_results = []
    for name, m in clf_fitted.items():
        tr  = m.score(X_tr_cs, y_tr_c)
        te  = m.score(X_te_cs, y_te_c)
        gap = tr - te
        if gap > 0.10:   status = '❌ Overfitting'
        elif gap > 0.05: status = '⚠️ Overfitting Ringan'
        elif te < 0.65:  status = '⚠️ Underfitting'
        else:            status = '✅ Fit Baik'
        bv_results.append({'Model': name, 'Train Acc': round(tr,4),
                           'Test Acc': round(te,4), 'Gap': round(gap,4), 'Status': status})
    bv_df = pd.DataFrame(bv_results)

    # Learning curves dihitung di sini agar tidak timeout saat render
    lc_data = {}
    for name, m in clf_fitted.items():
        n   = min(10000, X_tr_cs.shape[0])
        idx = np.random.RandomState(42).choice(X_tr_cs.shape[0], n, replace=False)
        sz, t_sc, v_sc = learning_curve(
            m, X_tr_cs[idx], y_tr_c.values[idx],
            cv=3, train_sizes=np.linspace(0.1, 1.0, 6),
            scoring='accuracy', n_jobs=-1
        )
        lc_data[name] = {'sz': sz, 'train': t_sc, 'val': v_sc}

    return {
        'df_clean'         : df_clean,
        'selected_features': selected_features,
        'scaler_c'         : scaler_c,
        'X_scaled'         : X_scaled,
        'sil_score'        : sil,
        'reg_df'           : reg_df,
        'clf_df'           : clf_df,
        'clf_fitted'       : clf_fitted,
        'bv_df'            : bv_df,
        'lc_data'          : lc_data,
        'X_tr_cs'          : X_tr_cs,
        'y_tr_c'           : y_tr_c,
        'X_te_cs'          : X_te_cs,
        'y_te_c'           : y_te_c,
    }

# ── Header ──
st.title("🏨 Hotel Booking Demand — Dashboard Analitik")
data = load_and_train()

df_clean          = data['df_clean']
selected_features = data['selected_features']
scaler_c          = data['scaler_c']
sil_score         = data['sil_score']
reg_df            = data['reg_df']
clf_df            = data['clf_df']
clf_fitted        = data['clf_fitted']
bv_df             = data['bv_df']
lc_data           = data['lc_data']

cluster_counts  = df_clean['cluster'].value_counts().sort_index()
cluster_profile = df_clean.groupby('cluster')[selected_features].mean().round(2)
c0 = df_clean[df_clean['cluster'] == 0]
c1 = df_clean[df_clean['cluster'] == 1]
label_0 = 'Premium / Long-stay' if c0['adr'].mean() > c1['adr'].mean() else 'Budget / Short-stay'
label_1 = 'Budget / Short-stay' if label_0 == 'Premium / Long-stay' else 'Premium / Long-stay'

# ── Sidebar ──
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman:", [
    "📊 Ringkasan Eksekutif",
    "🔵 Profil Cluster",
    "🤖 Prediksi Segmen Tamu",
    "📈 Perbandingan Model",
    "🔎 Overfitting & Underfitting",
])
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Info Dataset**
- Total booking: `{len(df_clean):,}`
- Fitur: `{len(selected_features)}`
- Cluster: `2`
- Silhouette Score: `{sil_score:.4f}`
""")

# ═══════════════════════════════════════════════════════
# PAGE 1: Ringkasan Eksekutif
# ═══════════════════════════════════════════════════════
if page == "📊 Ringkasan Eksekutif":
    st.header("📊 Ringkasan Eksekutif")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Booking", f"{len(df_clean):,}")
    col2.metric("Silhouette Score", f"{sil_score:.4f}")
    col3.metric("Akurasi Terbaik", f"{clf_df['Accuracy'].max():.1%}")
    col4.metric("AUC-ROC Terbaik", f"{clf_df['AUC-ROC'].max():.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Ringkasan Eksekutif — Hotel Booking Analysis', fontsize=14, fontweight='bold')

    axes[0,0].pie(cluster_counts,
                  labels=[f'Cluster {i}' for i in cluster_counts.index],
                  autopct='%1.1f%%', colors=['#2196F3','#FF9800'], startangle=90)
    axes[0,0].set_title('Distribusi Cluster Tamu')

    sns.boxplot(x='cluster', y='adr',
                data=df_clean.sample(5000, random_state=42),
                palette=['#2196F3','#FF9800'], ax=axes[0,1])
    axes[0,1].set_title('Distribusi ADR per Cluster')

    sns.boxplot(x='cluster', y='lead_time',
                data=df_clean.sample(5000, random_state=42),
                palette=['#2196F3','#FF9800'], ax=axes[0,2])
    axes[0,2].set_title('Lead Time per Cluster')

    reg_df.set_index('Model')[['MAE','RMSE']].plot(
        kind='bar', ax=axes[1,0], color=['#ef9a9a','#ffcc80'])
    axes[1,0].set_title('Performa Model Regresi (MAE & RMSE)')
    axes[1,0].tick_params(axis='x', rotation=15)

    clf_df.set_index('Model')[['Accuracy','F1-Score','AUC-ROC']].plot(
        kind='bar', ax=axes[1,1], color=['#a5d6a7','#80deea','#ce93d8'])
    axes[1,1].set_title('Performa Model Klasifikasi')
    axes[1,1].tick_params(axis='x', rotation=15)
    axes[1,1].set_ylim(0, 1.1)

    profile_norm = (cluster_profile - cluster_profile.min()) / \
                   (cluster_profile.max() - cluster_profile.min() + 1e-9)
    profile_norm.T.plot(kind='bar', ax=axes[1,2], color=['#2196F3','#FF9800'])
    axes[1,2].set_title('Profil Cluster (Normalized)')
    axes[1,2].tick_params(axis='x', rotation=30)
    axes[1,2].legend([f'Cluster {i}' for i in cluster_profile.index])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════
# PAGE 2: Profil Cluster
# ═══════════════════════════════════════════════════════
elif page == "🔵 Profil Cluster":
    st.header("🔵 Profil Tiap Cluster")

    col1, col2 = st.columns(2)
    for col, cl, label in [(col1, 0, label_0), (col2, 1, label_1)]:
        cdata = df_clean[df_clean['cluster'] == cl]
        with col:
            st.markdown(f"### Cluster {cl} — {label}")
            st.markdown(f"**{len(cdata):,} booking ({len(cdata)/len(df_clean)*100:.1f}%)**")
            for feat in selected_features:
                st.metric(feat, f"{cdata[feat].mean():.2f}", f"±{cdata[feat].std():.2f}")

    st.markdown("---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    profile_norm = (cluster_profile - cluster_profile.min()) / \
                   (cluster_profile.max() - cluster_profile.min() + 1e-9)
    for cl, color, ax_i in [(0,'#2196F3',0),(1,'#FF9800',1)]:
        axes[ax_i].bar(selected_features, profile_norm.loc[cl],
                       color=color, edgecolor='black', alpha=0.8)
        axes[ax_i].set_title(f'Profil Cluster {cl} (Normalized 0-1)')
        axes[ax_i].set_ylim(0, 1.1)
        axes[ax_i].tick_params(axis='x', rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Tabel Profil Cluster")
    st.dataframe(cluster_profile.T, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 3: Prediksi
# ═══════════════════════════════════════════════════════
elif page == "🤖 Prediksi Segmen Tamu":
    st.header("🤖 Prediksi Segmen Tamu Baru")

    col_form, col_result = st.columns([1, 1])
    with col_form:
        st.subheader("Input Parameter Tamu")
        adr          = st.slider('ADR (harga/malam)', 0.0, 500.0, 100.0, 5.0)
        lead_time    = st.slider('Lead Time (hari)', 0, 400, 30, 5)
        total_stays  = st.slider('Total Stays (malam)', 0, 20, 3, 1)
        total_guests = st.slider('Total Guests', 1, 10, 2, 1)
        special_req  = st.slider('Special Requests', 0, 5, 0, 1)
        prev_cancel  = st.slider('Previous Cancellations', 0, 10, 0, 1)
        predict_btn  = st.button('🔍 Prediksi Sekarang', type='primary', use_container_width=True)

    with col_result:
        st.subheader("Hasil Prediksi")
        if predict_btn:
            inp    = np.array([[adr, lead_time, total_stays, total_guests, special_req, prev_cancel]])
            inp_sc = scaler_c.transform(inp)

            votes, rows = [], []
            for name, model in clf_fitted.items():
                pred = model.predict(inp_sc)[0]
                prob = model.predict_proba(inp_sc)[0]
                votes.append(pred)
                lbl  = label_1 if pred == 1 else label_0
                rows.append({'Model': name, 'Prediksi': f'Cluster {pred} ({lbl})',
                             'P(C0)': f'{prob[0]:.1%}', 'P(C1)': f'{prob[1]:.1%}'})

            final = max(set(votes), key=votes.count)
            st.success(f"🏆 Cluster {final} — {label_1 if final==1 else label_0}")
            st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

            probs_rf = clf_fitted['Random Forest'].predict_proba(inp_sc)[0]
            fig, ax  = plt.subplots(figsize=(6, 2.5))
            bars = ax.barh([f'Cluster 0 ({label_0})', f'Cluster 1 ({label_1})'],
                           probs_rf, color=['#2196F3','#FF9800'], edgecolor='black')
            ax.set_xlim(0, 1)
            ax.set_title('Probabilitas (Random Forest)')
            for bar, p in zip(bars, probs_rf):
                ax.text(min(bar.get_width()+0.02, 0.92),
                        bar.get_y()+bar.get_height()/2,
                        f'{p:.1%}', va='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Atur parameter di kiri, lalu klik Prediksi Sekarang.")

# ═══════════════════════════════════════════════════════
# PAGE 4: Perbandingan Model
# ═══════════════════════════════════════════════════════
elif page == "📈 Perbandingan Model":
    st.header("📈 Perbandingan Performa Model")

    tab_reg, tab_clf = st.tabs(["Regresi", "Klasifikasi"])

    with tab_reg:
        st.dataframe(reg_df.set_index('Model'), use_container_width=True)
        best_r = reg_df.loc[reg_df['R² Score'].idxmax(), 'Model']
        st.success(f"🏆 Terbaik: **{best_r}** (R²={reg_df['R² Score'].max():.4f})")

        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(reg_df))
        w = 0.2
        ax.bar([i-w for i in x], reg_df['MAE'],       w, label='MAE',  color='#ef9a9a')
        ax.bar(list(x),          reg_df['RMSE'],      w, label='RMSE', color='#ffcc80')
        ax.bar([i+w for i in x], reg_df['R² Score'],  w, label='R²',   color='#a5d6a7')
        ax.set_xticks(list(x))
        ax.set_xticklabels(reg_df['Model'], rotation=10, ha='right')
        ax.set_title('Metrik Model Regresi')
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab_clf:
        st.dataframe(clf_df.set_index('Model'), use_container_width=True)
        best_c = clf_df.loc[clf_df['F1-Score'].idxmax(), 'Model']
        st.success(f"🏆 Terbaik: **{best_c}** (F1={clf_df['F1-Score'].max():.4f})")

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for name, model in clf_fitted.items():
            yprob = model.predict_proba(data['X_te_cs'])[:, 1]
            fpr, tpr, _ = roc_curve(data['y_te_c'], yprob)
            auc = roc_auc_score(data['y_te_c'], yprob)
            axes[0].plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f})')
        axes[0].plot([0,1],[0,1],'k--', lw=1)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend(fontsize=8)

        cm = confusion_matrix(data['y_te_c'], clf_fitted[best_c].predict(data['X_te_cs']))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                    xticklabels=['Cluster 0','Cluster 1'],
                    yticklabels=['Cluster 0','Cluster 1'])
        axes[1].set_title(f'Confusion Matrix — {best_c}')
        axes[1].set_ylabel('Aktual')
        axes[1].set_xlabel('Prediksi')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════════════════
# PAGE 5: Overfitting & Underfitting
# ═══════════════════════════════════════════════════════
elif page == "🔎 Overfitting & Underfitting":
    st.header("🔎 Analisis Overfitting & Underfitting")

    st.subheader("Train vs Test Score")
    st.dataframe(bv_df.set_index('Model'), use_container_width=True)

    ok = sum(1 for s in bv_df['Status'] if '✅' in s)
    if ok == len(bv_df):
        st.success(f"✅ Semua {len(bv_df)} model dikategorikan Fit Baik.")
    else:
        over = [r['Model'] for _, r in bv_df.iterrows() if '✅' not in r['Status']]
        st.warning(f"⚠️ Model berikut perlu perhatian: {', '.join(over)}")

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_g = ['#c6efce' if g<=0.05 else ('#ffeb9c' if g<=0.10 else '#ffc7ce')
                for g in bv_df['Gap']]
    bars = ax.bar(bv_df['Model'], bv_df['Gap'], color=colors_g, edgecolor='black')
    ax.axhline(0.05, color='orange', linestyle='--', lw=1.5, label='Overfitting ringan (0.05)')
    ax.axhline(0.10, color='red',    linestyle='--', lw=1.5, label='Overfitting (0.10)')
    for bar, gap in zip(bars, bv_df['Gap']):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f'{gap:+.4f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Gap (Train Acc − Test Acc) per Model')
    ax.set_ylabel('Gap Score')
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(bv_df['Gap'].max()*1.5, 0.15))
    plt.xticks(rotation=10, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Learning Curves")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, lc) in zip(axes, lc_data.items()):
        tm = lc['train'].mean(axis=1); ts = lc['train'].std(axis=1)
        vm = lc['val'].mean(axis=1);   vs = lc['val'].std(axis=1)
        ax.plot(lc['sz'], tm, 'o-', color='#2196F3', label='Train')
        ax.plot(lc['sz'], vm, 's-', color='#FF9800', label='Val')
        ax.fill_between(lc['sz'], tm-ts, tm+ts, alpha=0.12, color='#2196F3')
        ax.fill_between(lc['sz'], vm-vs, vm+vs, alpha=0.12, color='#FF9800')
        ax.set_title(f'{name}\nGap: {tm[-1]-vm[-1]:+.3f}', fontsize=9)
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("Train ≈ Val tinggi → Fit Baik ✅ | Train tinggi Val rendah → Overfitting ❌ | Keduanya rendah → Underfitting ⚠️")
