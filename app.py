# Update the Streamlit app to support MULTIPLE Excel uploads and combined analysis.
# It will overwrite /mnt/data/app.py and keep all previous features, adding multi-file concatenation,
# file-name filtering, and per-file summaries.

app_code = r'''
# -*- coding: utf-8 -*-
"""
🧪 Tetkik Analiz Arayüzü (Streamlit) — Çoklu Dosya Desteği
Yazar: Muammer
Çalıştırma:
    1) pip install streamlit pandas numpy scipy openpyxl matplotlib
    2) streamlit run app.py
Not: Grafikler matplotlib ile, seaborn kullanılmıyor.
"""

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tetkik Analiz Arayüzü", layout="wide")

# =============== Yardımcı Fonksiyonlar =============== #
REQ_COLS = ["PROTOKOL_NO", "TCKIMLIK_NO", "TETKIK_ISMI", "TEST_DEGERI", "CINSIYET"]

def coerce_numeric(series: pd.Series) -> pd.Series:
    """Virgüllü ondalıkları ve metinleri sayıya çevirir."""
    s = series.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
    s = pd.to_numeric(s, errors="coerce")
    return s

def check_columns(df: pd.DataFrame) -> list:
    missing = [c for c in REQ_COLS if c not in df.columns]
    return missing

def descr_stats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "q1": np.nan,
                "median": np.nan, "q3": np.nan, "max": np.nan, "cv%": np.nan, "iqr": np.nan}
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    cv = (np.std(x, ddof=1) / np.mean(x)) * 100 if np.mean(x) != 0 else np.nan
    return {
        "count": int(x.shape[0]),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "q1": float(q1),
        "median": float(np.median(x)),
        "q3": float(q3),
        "max": float(np.max(x)),
        "cv%": float(cv),
        "iqr": float(iqr),
    }

def normality_flag(x: pd.Series, alpha=0.05) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 3:
        return "yetersiz"
    try:
        if len(x) <= 5000:
            stat, p = stats.shapiro(x)
            return "normal" if p >= alpha else "non-normal"
        else:
            res = stats.anderson(x, dist="norm")
            crit = res.critical_values[2]  # 5% seviyesi
            return "normal" if res.statistic < crit else "non-normal"
    except Exception:
        return "bilinmiyor"

def nonparametric_test_by_group(df, val_col, grp_col):
    # Kaç grup?
    groups = [g.dropna() for _, g in df.groupby(grp_col)[val_col]]
    groups = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
    groups = [g for g in groups if len(g) > 0]
    unique_groups = df[grp_col].dropna().unique()
    unique_groups = [g for g in unique_groups if df[df[grp_col] == g][val_col].notna().sum() > 0]

    if len(unique_groups) < 2:
        return "Karşılaştırma için en az 2 grup gerekli.", None

    if len(unique_groups) == 2:
        # Mann-Whitney U
        gnames = list(unique_groups)
        x = pd.to_numeric(df[df[grp_col] == gnames[0]][val_col], errors="coerce").dropna()
        y = pd.to_numeric(df[df[grp_col] == gnames[1]][val_col], errors="coerce").dropna()
        if len(x) >= 1 and len(y) >= 1:
            stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            return f"Mann–Whitney U: U={stat:.2f}, p={p:.4g} ({gnames[0]} vs {gnames[1]})", ("MWU", stat, p, gnames[0], gnames[1])
        else:
            return "Gruplarda yeterli gözlem yok.", None
    else:
        # Kruskal-Wallis
        stat, p = stats.kruskal(*groups)
        return f"Kruskal–Wallis: H={stat:.2f}, p={p:.4g} (grup sayısı: {len(unique_groups)})", ("KW", stat, p, unique_groups)

def make_boxplot(df, x_col, y_col, title="Kutu Grafiği"):
    valid = df[[x_col, y_col]].copy()
    valid[y_col] = pd.to_numeric(valid[y_col], errors="coerce")
    valid = valid.dropna()
    if valid.empty:
        st.info("Grafik için yeterli veri yok.")
        return
    cats = list(valid[x_col].astype(str).unique())
    data = [valid[valid[x_col].astype(str) == c][y_col].values for c in cats]
    fig, ax = plt.subplots()
    ax.boxplot(data, labels=cats, showmeans=True)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)

def make_hist(df, col, bins=30, title="Histogram"):
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if x.empty:
        st.info("Histogram için yeterli veri yok.")
        return
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Frekans")
    st.pyplot(fig)

def export_df(df, name="export.csv"):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ CSV indir", data=csv, file_name=name, mime="text/csv")

# =============== Arayüz =============== #
st.title("🧪 Tetkik Analiz Arayüzü — Çoklu Dosya")
st.caption("Sütunlar: PROTOKOL_NO, TCKIMLIK_NO, TETKIK_ISMI, TEST_DEGERI, CINSIYET — Birden çok Excel dosyasını seçip üst üste analiz edebilirsiniz.")

uploads = st.file_uploader("Excel dosyaları yükleyin (.xlsx, .xls) — Çoklu seçim yapın", type=["xlsx", "xls"], accept_multiple_files=True)

if not uploads:
    st.info("Birden fazla dosyayı aynı anda seçebilir veya yüklemeyi tekrarlayıp ekleyebilirsiniz (Streamlit oturumunda seçtikleriniz tutulur).")
    st.stop()

frames = []
skipped = []
for upl in uploads:
    try:
        tmp = pd.read_excel(upl)
        missing = check_columns(tmp)
        if missing:
            skipped.append((upl.name, f"Eksik sütun: {missing}"))
            continue
        tmp["SOURCE_FILE"] = upl.name
        frames.append(tmp)
    except Exception as e:
        skipped.append((upl.name, f"Okuma hatası: {e}"))

if skipped:
    for nm, msg in skipped:
        st.warning(f"'{nm}' atlandı → {msg}")

if not frames:
    st.error("Yüklenen dosyaların hiçbirinden uygun veri okunamadı.")
    st.stop()

df = pd.concat(frames, ignore_index=True)

# Sayısal dönüştürme
df["TEST_DEGERI"] = coerce_numeric(df["TEST_DEGERI"])

# Filtreler
left, right = st.columns([3, 2])
with left:
    # Tetkik seçimi
    unique_tests = sorted([str(x) for x in df["TETKIK_ISMI"].dropna().unique()])
    selected_tests = st.multiselect("Analiz edilecek tetkikler", options=unique_tests, default=unique_tests[:1])
with right:
    # Cinsiyet filtresi (opsiyonel)
    sexes = [str(x) for x in df["CINSIYET"].dropna().unique()]
    chosen_sex = st.multiselect("Cinsiyet filtresi (opsiyonel)", options=sexes, default=sexes)
    # Dosya filtresi (opsiyonel)
    files = [str(x) for x in df["SOURCE_FILE"].dropna().unique()]
    chosen_files = st.multiselect("Dosya filtresi (opsiyonel)", options=files, default=files)

# Veri alt kümesi
work = df.copy()
if chosen_sex:
    work = work[work["CINSIYET"].astype(str).isin(chosen_sex)]
if chosen_files:
    work = work[work["SOURCE_FILE"].astype(str).isin(chosen_files)]
if selected_tests:
    work = work[work["TETKIK_ISMI"].astype(str).isin(selected_tests)]

st.subheader("🔎 Genel Bilgiler (Birleştirilmiş Veri)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Toplam Satır", f"{len(df):,}")
c2.metric("Benzersiz TCKIMLIK_NO", f"{df['TCKIMLIK_NO'].nunique():,}")
c3.metric("Benzersiz Tetkik", f"{df['TETKIK_ISMI'].nunique():,}")
c4.metric("Benzersiz Cinsiyet", f"{df['CINSIYET'].nunique():,}")
c5.metric("Dosya Sayısı", f"{df['SOURCE_FILE'].nunique():,}")

st.write("Seçimler sonrası kalan satır sayısı:", len(work))

# =============== Dosya Bazında Özet (Opsiyonel) =============== #
with st.expander("📦 Dosya Bazında Özet (N, tetkik sayısı, hasta sayısı)"):
    per_file = df.groupby("SOURCE_FILE").agg(
        N=("PROTOKOL_NO", "size"),
        Hasta_Sayisi=("TCKIMLIK_NO", "nunique"),
        Tetkik_Sayisi=("TETKIK_ISMI", "nunique")
    ).reset_index()
    st.dataframe(per_file, use_container_width=True)
    export_df(per_file, "dosya_bazinda_ozet.csv")

# =============== Her Tetkik İçin Ayrı Analiz =============== #
st.header("📊 Tetkik Bazlı Analiz (Birleştirilmiş + Filtreli)")

results_rows = []

for test_name in selected_tests:
    sub = work[work["TETKIK_ISMI"].astype(str) == test_name].copy()
    if sub.empty:
        continue

    st.subheader(f"🧷 {test_name}")

    # Tanımlayıcı istatistikler (genel)
    stats_overall = descr_stats(sub["TEST_DEGERI"])
    normal_flag = normality_flag(sub["TEST_DEGERI"])

    # Cinsiyet kırılımı
    by_sex = sub.groupby("CINSIYET", dropna=False)["TEST_DEGERI"].apply(descr_stats).apply(pd.Series).reset_index()

    # Kaynak dosyaya göre kırılım
    by_file = sub.groupby("SOURCE_FILE", dropna=False)["TEST_DEGERI"].apply(descr_stats).apply(pd.Series).reset_index()

    # Karşılaştırma testi
    msg, test_info = nonparametric_test_by_group(sub, "TEST_DEGERI", "CINSIYET")

    # Özet tablo satırı
    results_rows.append({
        "TETKIK_ISMI": test_name,
        "N": stats_overall["count"],
        "Mean": stats_overall["mean"],
        "Median": stats_overall["median"],
        "Std": stats_overall["std"],
        "Min": stats_overall["min"],
        "Q1": stats_overall["q1"],
        "Q3": stats_overall["q3"],
        "Max": stats_overall["max"],
        "Normalite": normal_flag,
        "Test": msg
    })

    # Gösterimler
    tabs = st.tabs(["Tanımlayıcı", "Cinsiyet Kırılımı", "Dosya Kırılımı", "İstatistiksel Test", "Histogram", "Boxplot (Cinsiyete göre)"])
    with tabs[0]:
        st.write("**Genel Tanımlayıcı İstatistikler**")
        st.table(pd.DataFrame([stats_overall]))
    with tabs[1]:
        st.write("**Cinsiyete Göre Tanımlayıcılar**")
        st.dataframe(by_sex, use_container_width=True)
    with tabs[2]:
        st.write("**Kaynak Dosyaya Göre Tanımlayıcılar**")
        st.dataframe(by_file, use_container_width=True)
    with tabs[3]:
        st.write("**Karşılaştırma (Nonparametrik)**")
        st.info(msg)
    with tabs[4]:
        make_hist(sub, "TEST_DEGERI", bins=30, title=f"{test_name} - Histogram")
    with tabs[5]:
        make_boxplot(sub, "CINSIYET", "TEST_DEGERI", title=f"{test_name} - Cinsiyete Göre Boxplot")

# Toplu özet
if results_rows:
    st.header("🧾 Toplu Özet Tablosu (Seçili Tetkikler)")
    res_df = pd.DataFrame(results_rows)
    st.dataframe(res_df, use_container_width=True)
    export_df(res_df, name="tetkik_ozet.csv")

# =============== Tüm Tetkikler için Otomatik Rapor =============== #
st.header("📑 Otomatik Rapor (Tüm Tetkikler, Birleştirilmiş Veri)")
if st.button("Tüm tetkikler için raporu üret"):
    rows = []
    for t in sorted(df["TETKIK_ISMI"].dropna().astype(str).unique()):
        sub = df[df["TETKIK_ISMI"].astype(str) == t].copy()
        sub["TEST_DEGERI"] = pd.to_numeric(sub["TEST_DEGERI"], errors="coerce")
        sub = sub.dropna(subset=["TEST_DEGERI"])
        if sub.empty:
            continue
        stats_overall = descr_stats(sub["TEST_DEGERI"])
        msg, test_info = nonparametric_test_by_group(sub, "TEST_DEGERI", "CINSIYET")
        rows.append({
            "TETKIK_ISMI": t,
            "N": stats_overall["count"],
            "Mean": stats_overall["mean"],
            "Median": stats_overall["median"],
            "Std": stats_overall["std"],
            "Min": stats_overall["min"],
            "Q1": stats_overall["q1"],
            "Q3": stats_overall["q3"],
            "Max": stats_overall["max"],
            "Normalite": normality_flag(sub["TEST_DEGERI"]),
            "Test": msg
        })
    if rows:
        rpt = pd.DataFrame(rows)
        st.success("Rapor hazırlandı.")
        st.dataframe(rpt, use_container_width=True)
        export_df(rpt, name="tum_tetkikler_rapor.csv")
    else:
        st.warning("Rapor için uygun veri bulunamadı.")

# =============== Ek Araçlar =============== #
st.header("🧰 Ek Araçlar")
with st.expander("Pivot: TCKIMLIK_NO × Tetkik sayıları (Birleştirilmiş)"):
    pivot = (df
             .assign(has_val=df["TEST_DEGERI"].notna().astype(int))
             .pivot_table(index="TCKIMLIK_NO", columns="TETKIK_ISMI", values="has_val",
                          aggfunc="sum", fill_value=0))
    st.dataframe(pivot)
    export_df(pivot.reset_index(), name="pivot_tckimlik_tetkik.csv")

with st.expander("Ham Veri Ön İzleme (İlk 200 satır)"):
    st.dataframe(df.head(200))

st.caption("Not: İki grup varsa Mann–Whitney U; 3+ grup varsa Kruskal–Wallis uygulanır. Normalite bilgilendirme amaçlıdır. 'SOURCE_FILE' sütunu hangi dosyadan geldiğini gösterir.")
'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

'/mnt/data/app.py'
