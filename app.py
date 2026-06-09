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

    # ── 1. Hapus Duplikat ──
    df_clean = df.copy()
    df_clean.drop_duplicates(inplace=True)

    # ── 2. Imputasi Missing Values ──
    df_clean['children'].fillna(0, inplace=True)
    df_clean['country'].fillna(df_clean['country'].mode()[0], inplace=True)
    df_clean['agent'].fillna(0, inplace=True)
    df_clean['company'].fillna(0, inplace=True)

    # ── 3. Hapus Booking Tanpa Tamu ──
    df_clean = df_clean[(df_clean['adults'] + df_clean['children'] + df_clean['babies']) > 0]

    # ── 4. Feature Engineering ──
    df_clean['total_stays']   = df_clean['stays_in_weekend_nights'] + df_clean['stays_in_week_nights']
    df_clean['total_guests']  = df_clean['adults'] + df_clean['children'] + df_clean['babies']
    df_clean['total_revenue'] = df_clean['adr'] * df_clean['total_stays']
    month_map = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
                 'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
    df_clean['arrival_month_num'] = df_clean['arrival_date_month'].map(month_map)

    # ── 5. Deteksi & Hapus Outlier dengan IQR (adr & lead_time) ──
    for feat in ['adr', 'lead_time']:
        Q1    = df_clean[feat].quantile(0.25)
        Q3    = df_clean[feat].quantile(0.75)
        IQR   = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        df_clean = df_clean[(df_clean[feat] >= lower) & (df_clean[feat] <= upper)]
    df_clean = df_clean[df_clean['adr'] >= 0]

    # ── Data Selection ──
    selected_features = [
        'adr', 'lead_time', 'total_stays', 'total_guests',
        'total_of_special_requests', 'previous_cancellations'
    ]

    # ── Clustering ──
    X_cluster = df_clean[selected_features].copy()
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X_cluster)
    kmeans    = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
    cluster_labels       = kmeans.fit_predict(X_scaled)
    df_clean['cluster']  = cluster_labels
    sil = silhouette_score(X_scaled, cluster_labels)

    # ── Regression ──
    X       = df_clean[selected_features].copy()
    y_reg   = df_clean['cluster'].astype(float)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42)
    scaler_r   = StandardScaler()
    X_train_rs = scaler_r.fit_transform(X_train_r)
    X_test_rs  = scaler_r.transform(X_test_r)

    models_reg = {
        'Linear Regression'       : LinearRegression(),
        'Random Forest Regressor' : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    }
    reg_results = []
    reg_preds   = {}
    for name, model in models_reg.items():
        model.fit(X_train_rs, y_train_r)
        y_pred = model.predict(X_test_rs)
        reg_preds[name] = y_pred
        reg_results.append({
            'Model'   : name,
            'MAE'     : round(mean_absolute_error(y_train_r, model.predict(X_train_rs)) if False else mean_absolute_error(y_test_r, y_pred), 4),
            'MSE'     : round(mean_squared_error(y_test_r, y_pred), 4),
            'RMSE'    : round(np.sqrt(mean_squared_error(y_test_r, y_pred)), 4),
            'R² Score': round(r2_score(y_test_r, y_pred), 4)
        })
    reg_df        = pd.DataFrame(reg_results)
    best_reg_name = reg_df.loc[reg_df['R² Score'].idxmax(), 'Model']

    # ── Classification ──
    y_clf = df_clean['cluster'].astype(int)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    scaler_c   = StandardScaler()
    X_train_cs = scaler_c.fit_transform(X_train_c)
    X_test_cs  = scaler_c.transform(X_test_c)

    models_clf = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes'        : GaussianNB(),
        'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    }
    clf_results       = []
    clf_models_fitted = {}
    for name, model in models_clf.items():
        model.fit(X_train_cs, y_train_c)
        y_pred = model.predict(X_test_cs)
        y_prob = model.predict_proba(X_test_cs)[:, 1]
        clf_results.append({
            'Model'    : name,
            'Accuracy' : round(accuracy_score(y_test_c, y_pred), 4),
            'Precision': round(precision_score(y_test_c, y_pred), 4),
            'Recall'   : round(recall_score(y_test_c, y_pred), 4),
            'F1-Score' : round(f1_score(y_test_c, y_pred), 4),
            'AUC-ROC'  : round(roc_auc_score(y_test_c, y_prob), 4)
        })
        clf_models_fitted[name] = model
    clf_df        = pd.DataFrame(clf_results)
    best_clf_name = clf_df.loc[clf_df['F1-Score'].idxmax(), 'Model']

    # ── Bias-Variance (Overfitting) ──
    bv_results = []
    for name, model in clf_models_fitted.items():
        tr  = model.score(X_train_cs, y_train_c)
        te  = model.score(X_test_cs,  y_test_c)
        gap = tr - te
        if gap > 0.10:   status = '❌ Overfitting'
        elif gap > 0.05: status = '⚠️ Overfitting Ringan'
        elif te < 0.65:  status = '⚠️ Underfitting'
        else:            status = '✅ Fit Baik'
        bv_results.append({'Model': name, 'Train Acc': round(tr,4),
                           'Test Acc': round(te,4), 'Gap': round(gap,4), 'Status': status})
    bv_df = pd.DataFrame(bv_results)

    # ── Learning Curves (dihitung di cache agar tidak timeout) ──
    lc_data = {}
    for name, model in clf_models_fitted.items():
        n   = min(10000, X_train_cs.shape[0])
        idx = np.random.RandomState(42).choice(X_train_cs.shape[0], n, replace=False)
        sz, t_sc, v_sc = learning_curve(
            model, X_train_cs[idx], y_train_c.values[idx],
            cv=3, train_sizes=np.linspace(0.1, 1.0, 6),
            scoring='accuracy', n_jobs=1
        )
        lc_data[name] = {'sz': sz, 'train': t_sc, 'val': v_sc}

    return {
        'df_clean'         : df_clean,
        'selected_features': selected_features,
        'scaler_c'         : scaler_c,
        'X_scaled'         : X_scaled,
        'sil_score'        : sil,
        'kmeans'           : kmeans,
        'reg_df'           : reg_df,
        'best_reg_name'    : best_reg_name,
        'reg_preds'        : reg_preds,
        'models_reg'       : models_reg,
        'X_test_rs'        : X_test_rs,
        'y_test_r'         : y_test_r,
        'clf_df'           : clf_df,
        'best_clf_name'    : best_clf_name,
        'clf_models_fitted': clf_models_fitted,
        'bv_df'            : bv_df,
        'lc_data'          : lc_data,
        'X_train_cs'       : X_train_cs,
        'X_test_cs'        : X_test_cs,
        'y_train_c'        : y_train_c,
        'y_test_c'         : y_test_c,
    }

# ── Header ──
st.title("🏨 Hotel Booking Demand — Dashboard Analitik")
data = load_and_train()

df_clean           = data['df_clean']
selected_features  = data['selected_features']
scaler_c           = data['scaler_c']
sil_score          = data['sil_score']
reg_df             = data['reg_df']
best_reg_name      = data['best_reg_name']
clf_df             = data['clf_df']
best_clf_name      = data['best_clf_name']
clf_models_fitted  = data['clf_models_fitted']
bv_df              = data['bv_df']
lc_data            = data['lc_data']

cluster_counts  = df_clean['cluster'].value_counts().sort_index()
cluster_profile = df_clean.groupby('cluster')[selected_features].mean().round(2)

# Label cluster sesuai notebook: Cluster 0 = Budget, Cluster 1 = Premium
label_0 = 'Budget / Short-stay'
label_1 = 'Premium / Long-stay'

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
    axes[0,1].set_xlabel('Cluster')

    sns.boxplot(x='cluster', y='lead_time',
                data=df_clean.sample(5000, random_state=42),
                palette=['#2196F3','#FF9800'], ax=axes[0,2])
    axes[0,2].set_title('Lead Time per Cluster')
    axes[0,2].set_xlabel('Cluster')

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
    profile_norm = (cluster_profile - cluster_profile.min()) / \
                   (cluster_profile.max() - cluster_profile.min() + 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
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
            for name, model in clf_models_fitted.items():
                pred = model.predict(inp_sc)[0]
                prob = model.predict_proba(inp_sc)[0]
                votes.append(pred)
                lbl  = label_1 if pred == 1 else label_0
                rows.append({'Model': name, 'Prediksi': f'Cluster {pred} ({lbl})',
                             'P(Cluster 0)': f'{prob[0]:.1%}', 'P(Cluster 1)': f'{prob[1]:.1%}'})

            final = max(set(votes), key=votes.count)
            st.success(f"🏆 Prediksi Final (majority vote): Cluster {final} — {label_1 if final==1 else label_0}")
            st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

            # Gunakan model terbaik (Logistic Regression) untuk bar chart
            probs_best = clf_models_fitted[best_clf_name].predict_proba(inp_sc)[0]
            fig, ax = plt.subplots(figsize=(6, 2.5))
            bars = ax.barh([f'Cluster 0 ({label_0})', f'Cluster 1 ({label_1})'],
                           probs_best, color=['#2196F3','#FF9800'], edgecolor='black')
            ax.set_xlim(0, 1)
            ax.set_title(f'Probabilitas Prediksi ({best_clf_name})')
            ax.set_xlabel('Probabilitas')
            for bar, p in zip(bars, probs_best):
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
        st.success(f"🏆 Model terbaik: **{best_reg_name}** (R²={reg_df.loc[reg_df['Model']==best_reg_name,'R² Score'].values[0]:.4f})")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = range(len(reg_df))
        w = 0.2
        axes[0].bar([i-w for i in x], reg_df['MAE'],       w, label='MAE',  color='#ef9a9a')
        axes[0].bar(list(x),          reg_df['RMSE'],      w, label='RMSE', color='#ffcc80')
        axes[0].bar([i+w for i in x], reg_df['R² Score'],  w, label='R²',   color='#a5d6a7')
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(reg_df['Model'], rotation=10, ha='right')
        axes[0].set_title('Metrik Model Regresi')
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)

        # Aktual vs Prediksi (model terbaik)
        best_reg_model = data['models_reg'][best_reg_name]
        y_pred_best_r  = data['reg_preds'][best_reg_name]
        axes[1].scatter(data['y_test_r'], y_pred_best_r, alpha=0.2, color='steelblue', s=8)
        axes[1].plot([0,1],[0,1],'r--', lw=2)
        axes[1].set_xlabel('Aktual Cluster Label')
        axes[1].set_ylabel('Prediksi (nilai kontinu)')
        axes[1].set_title(f'Aktual vs Prediksi — {best_reg_name}')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab_clf:
        st.dataframe(clf_df.set_index('Model'), use_container_width=True)
        st.success(f"🏆 Model terbaik: **{best_clf_name}** (F1={clf_df.loc[clf_df['Model']==best_clf_name,'F1-Score'].values[0]:.4f})")

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Confusion Matrix — model terbaik (sesuai notebook)
        y_pred_best_c = clf_models_fitted[best_clf_name].predict(data['X_test_cs'])
        cm = confusion_matrix(data['y_test_c'], y_pred_best_c)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Cluster 0','Cluster 1'],
                    yticklabels=['Cluster 0','Cluster 1'])
        axes[0].set_title(f'Confusion Matrix — {best_clf_name}')
        axes[0].set_ylabel('Aktual')
        axes[0].set_xlabel('Prediksi')

        # ROC Curve semua model
        for name, model in clf_models_fitted.items():
            y_p = model.predict_proba(data['X_test_cs'])[:, 1]
            fpr, tpr, _ = roc_curve(data['y_test_c'], y_p)
            auc_val = roc_auc_score(data['y_test_c'], y_p)
            axes[1].plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc_val:.3f})')
        axes[1].plot([0,1],[0,1],'k--', linewidth=1, label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve — Semua Model Klasifikasi')
        axes[1].legend(loc='lower right', fontsize=9)
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
        st.success(f"✅ Semua {len(bv_df)} model dikategorikan Fit Baik — tidak ada overfitting/underfitting signifikan.")
    else:
        over = [r['Model'] for _, r in bv_df.iterrows() if '✅' not in r['Status']]
        st.warning(f"⚠️ Model berikut perlu perhatian: {', '.join(over)}")

    # Bar chart gap
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_g = ['#c6efce' if g<=0.05 else ('#ffeb9c' if g<=0.10 else '#ffc7ce')
                for g in bv_df['Gap']]
    bars = ax.bar(bv_df['Model'], bv_df['Gap'], color=colors_g, edgecolor='black')
    ax.axhline(0.05, color='orange', linestyle='--', lw=1.5, label='Overfitting ringan (0.05)')
    ax.axhline(0.10, color='red',    linestyle='--', lw=1.5, label='Overfitting (0.10)')
    for bar, gap in zip(bars, bv_df['Gap']):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f'{gap:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Gap (Train Acc − Test Acc) per Model')
    ax.set_ylabel('Gap Score')
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(bv_df['Gap'].max()*1.5, 0.15))
    plt.xticks(rotation=10, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Learning Curves
    st.subheader("Learning Curves")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Learning Curves — Deteksi Overfitting & Underfitting', fontsize=13, fontweight='bold')
    for ax, (name, lc) in zip(axes, lc_data.items()):
        tm = lc['train'].mean(axis=1); ts = lc['train'].std(axis=1)
        vm = lc['val'].mean(axis=1);   vs = lc['val'].std(axis=1)
        ax.plot(lc['sz'], tm, 'o-', color='#2196F3', label='Train Score')
        ax.plot(lc['sz'], vm, 's-', color='#FF9800', label='Val Score')
        ax.fill_between(lc['sz'], tm-ts, tm+ts, alpha=0.12, color='#2196F3')
        ax.fill_between(lc['sz'], vm-vs, vm+vs, alpha=0.12, color='#FF9800')
        ax.axhline(0.5, color='red', linestyle=':', linewidth=1, label='Baseline (0.5)')
        final_gap = tm[-1] - vm[-1]
        ax.set_title(f'{name}\nGap akhir: {final_gap:+.3f}', fontsize=10)
        ax.set_xlabel('Ukuran Training Set')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("Train ≈ Val tinggi → Fit Baik ✅ | Train tinggi Val rendah → Overfitting ❌ | Keduanya rendah → Underfitting ⚠️")
