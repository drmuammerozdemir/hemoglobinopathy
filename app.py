# Create an optimized version of the Streamlit app with performance improvements for many files.
# Key upgrades:
# - Cached, parallel file reading
# - Optional Polars backend (if installed)
# - Downcasting dtypes + categorical conversion
# - On-demand charts (to avoid rendering overhead)
# - Vectorized per-group stats via pandas agg
# - Heavier ops hidden behind expanders
# - Safer large-data display (sample/limit)

app_code = r'''
# -*- coding: utf-8 -*-
"""
🧪 Tetkik Analiz Arayüzü — Çoklu Dosya (Optimize)
- Çoklu dosya için hızlı okuma (paralel + cache)
- Bellek optimizasyonu (downcast, categorical)
- İsteğe bağlı Polars hızlandırma (kuruluysa)
- Büyük tabloları temkinli gösterim (örnekleme/limit)
- Grafikler isteğe bağlı oluşturulur (butonlar/checkbox), matplotlib (renk set edilmez)
Kurulum:
    pip install streamlit pandas numpy scipy openpyxl matplotlib
(opsiyonel hızlandırma)
    pip install polars pyarrow
Çalıştırma:
    streamlit run app_optimized.py
"""

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# ============== Ayarlar ============== #
st.set_page_config(page_title="Tetkik Analiz — Optimize", layout="wide")
REQ_COLS = ["PROTOKOL_NO", "TCKIMLIK_NO", "TETKIK_ISMI", "TEST_DEGERI", "CINSIYET"]
DISPLAY_LIMIT = 200  # Büyük veri için önizleme limiti

# Polars mevcut mu?
try:
    import polars as pl
    HAS_POLARS = True
except Exception:
    HAS_POLARS = False

# ============== Yardımcılar ============== #
def coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def check_columns(df: pd.DataFrame):
    return [c for c in REQ_COLS if c not in df.columns]

def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    # PROTOKOL_NO, TCKIMLIK_NO sayıya dönmesin (ID olabilir), diğer uygun alanları küçült
    # TEST_DEGERI numerik
    if "TEST_DEGERI" in df.columns:
        df["TEST_DEGERI"] = coerce_numeric(df["TEST_DEGERI"])
        df["TEST_DEGERI"] = pd.to_numeric(df["TEST_DEGERI"], errors="coerce", downcast="float")
    # Kategorik alanlar
    for col in ["TETKIK_ISMI", "CINSIYET", "SOURCE_FILE"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

def descr_stats_fast(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce")
    x = x[~x.isna()]
    if x.empty:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan,
                "q1": np.nan, "median": np.nan, "q3": np.nan, "max": np.nan,
                "cv%": np.nan, "iqr": np.nan}
    q = np.percentile(x, [25, 50, 75])
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    cv = (std / mean) * 100 if mean != 0 else np.nan
    return {
        "count": int(x.size),
        "mean": mean,
        "std": std,
        "min": float(x.min()),
        "q1": float(q[0]),
        "median": float(q[1]),
        "q3": float(q[2]),
        "max": float(x.max()),
        "cv%": float(cv),
        "iqr": float(q[2] - q[0]),
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
            crit = res.critical_values[2]  # 5%
            return "normal" if res.statistic < crit else "non-normal"
    except Exception:
        return "bilinmiyor"

def nonparametric_test_by_group(df, val_col, grp_col):
    # Gruplar
    groups = [g.dropna() for _, g in df.groupby(grp_col)[val_col]]
    groups = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
    groups = [g for g in groups if len(g) > 0]
    unique_groups = df[grp_col].dropna().unique()
    unique_groups = [g for g in unique_groups if df[df[grp_col] == g][val_col].notna().sum() > 0]

    if len(unique_groups) < 2:
        return "Karşılaştırma için en az 2 grup gerekli.", None

    if len(unique_groups) == 2:
        gnames = list(unique_groups)
        x = pd.to_numeric(df[df[grp_col] == gnames[0]][val_col], errors="coerce").dropna()
        y = pd.to_numeric(df[df[grp_col] == gnames[1]][val_col], errors="coerce").dropna()
        if len(x) >= 1 and len(y) >= 1:
            stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            return f"Mann–Whitney U: U={stat:.2f}, p={p:.4g} ({gnames[0]} vs {gnames[1]})", ("MWU", stat, p, gnames[0], gnames[1])
        else:
            return "Gruplarda yeterli gözlem yok.", None
    else:
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

# ============== Cache'li Dosya Okuma ============== #
@st.cache_data(show_spinner=False)
def read_one_excel_cached(file_bytes: bytes, engine_hint: str = "openpyxl") -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    return pd.read_excel(bio, engine=engine_hint)

def read_many_excels(files):
    # Paralel okuma
    def _read(upl):
        try:
            data = upl.read()  # bytes
            df = read_one_excel_cached(data)
            return (upl.name, df, None)
        except Exception as e:
            return (upl.name, None, str(e))

    out = []
    with ThreadPoolExecutor(max_workers=min(8, len(files))) as ex:
        for name, df, err in ex.map(_read, files):
            out.append((name, df, err))
    return out

# ============== UI Başlangıç ============== #
st.title("⚡ Tetkik Analiz — Çoklu Dosya (Optimize)")
st.caption("Büyük veri ve çoklu dosya için hızlandırılmış sürüm.")

uploads = st.file_uploader("Excel dosyaları (.xlsx, .xls) — Çoklu seçim", type=["xlsx", "xls"], accept_multiple_files=True)

use_polars = st.checkbox("Polars hızlandırmayı dene (kuruluysa)", value=HAS_POLARS and True,
                         help="Polars kurulu değilse otomatik devre dışı kalır.")

if not uploads:
    st.info("Birden çok dosyayı aynı anda seçin. Örn: 12 dosya.")
    st.stop()

with st.spinner("Dosyalar okunuyor..."):
    results = read_many_excels(uploads)

frames = []
skipped = []
for name, tmp, err in results:
    if err:
        skipped.append((name, f"Okuma hatası: {err}"))
        continue
    miss = check_columns(tmp)
    if miss:
        skipped.append((name, f"Eksik sütun: {miss}"))
        continue
    tmp["SOURCE_FILE"] = name
    frames.append(tmp)

if skipped:
    for nm, msg in skipped:
        st.warning(f"'{nm}' atlandı → {msg}")

if not frames:
    st.error("Uygun veri içeren dosya bulunamadı.")
    st.stop()

# Birleştir
df = pd.concat(frames, ignore_index=True)
df = downcast_df(df)

# Polars'a çevir (opsiyonel)
if use_polars and HAS_POLARS:
    try:
        pl_df = pl.from_pandas(df)
    except Exception:
        use_polars = False
        pl_df = None
else:
    pl_df = None

# ================= Filtreler ================= #
left, right = st.columns([3, 2])
with left:
    unique_tests = sorted([str(x) for x in df["TETKIK_ISMI"].dropna().unique()])
    default_pick = unique_tests[:5] if len(unique_tests) > 5 else unique_tests[:1]
    selected_tests = st.multiselect("Analiz edilecek tetkikler", options=unique_tests, default=default_pick)
with right:
    sexes = [str(x) for x in df["CINSIYET"].dropna().unique()]
    chosen_sex = st.multiselect("Cinsiyet filtresi", options=sexes, default=sexes)
    files = [str(x) for x in df["SOURCE_FILE"].dropna().unique()]
    chosen_files = st.multiselect("Dosya filtresi", options=files, default=files)

work = df.copy()
if chosen_sex:
    work = work[work["CINSIYET"].astype(str).isin(chosen_sex)]
if chosen_files:
    work = work[work["SOURCE_FILE"].astype(str).isin(chosen_files)]
if selected_tests:
    work = work[work["TETKIK_ISMI"].astype(str).isin(selected_tests)]

# ================= Genel Bilgiler ================= #
st.subheader("🔎 Genel Bilgiler (Birleştirilmiş)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Toplam Satır", f"{len(df):,}")
c2.metric("Benzersiz TCKIMLIK_NO", f"{df['TCKIMLIK_NO'].nunique():,}")
c3.metric("Benzersiz Tetkik", f"{df['TETKIK_ISMI'].nunique():,}")
c4.metric("Benzersiz Cinsiyet", f"{df['CINSIYET'].nunique():,}")
c5.metric("Dosya Sayısı", f"{df['SOURCE_FILE'].nunique():,}")

with st.expander("Ham Veri Ön İzleme (limitli)"):
    st.dataframe(work.head(DISPLAY_LIMIT), use_container_width=True)
    st.caption(f"Yalnızca ilk {DISPLAY_LIMIT} satır görüntülenir.")

# ================= Hızlı Özetler ================= #
st.header("⚙️ Hızlı Özet ve Kırılımlar")
colA, colB = st.columns(2)
with colA:
    st.write("**Cinsiyete Göre Tanımlayıcılar (Seçimdeki veri)**")
    if use_polars and pl_df is not None:
        pl_sub = pl.from_pandas(work[["CINSIYET", "TEST_DEGERI"]].copy())
        grp = (pl_sub
               .groupby("CINSIYET")
               .agg([pl.len().alias("count"),
                     pl.col("TEST_DEGERI").mean().alias("mean"),
                     pl.col("TEST_DEGERI").std().alias("std"),
                     pl.col("TEST_DEGERI").min().alias("min"),
                     pl.col("TEST_DEGERI").median().alias("median"),
                     pl.col("TEST_DEGERI").quantile(0.25, "nearest").alias("q1"),
                     pl.col("TEST_DEGERI").quantile(0.75, "nearest").alias("q3"),
                     pl.col("TEST_DEGERI").max().alias("max")])
               .to_pandas())
        st.dataframe(grp, use_container_width=True)
    else:
        grp = (work.groupby("CINSIYET", dropna=False)["TEST_DEGERI"]
               .agg(["count", "mean", "std", "min", "median", "max"]).reset_index())
        st.dataframe(grp, use_container_width=True)

with colB:
    st.write("**Dosyaya Göre Satır & Hasta & Tetkik Sayısı**")
    per_file = work.groupby("SOURCE_FILE").agg(
        N=("PROTOKOL_NO", "size"),
        Hasta_Sayisi=("TCKIMLIK_NO", "nunique"),
        Tetkik_Sayisi=("TETKIK_ISMI", "nunique")
    ).reset_index()
    st.dataframe(per_file, use_container_width=True)
    export_df(per_file, "dosya_bazinda_ozet_filtreli.csv")

# ================= Tetkik Bazlı Analiz ================= #
st.header("📊 Tetkik Bazlı Analiz (Seçim)")

results_rows = []

for test_name in selected_tests:
    sub = work[work["TETKIK_ISMI"].astype(str) == test_name].copy()
    if sub.empty:
        continue

    st.subheader(f"🧷 {test_name}")

    # Tanımlayıcılar
    stats_overall = descr_stats_fast(sub["TEST_DEGERI"])
    normal_flag = normality_flag(sub["TEST_DEGERI"])

    # Cinsiyet kırılımı (vektörize)
    by_sex = (sub.groupby("CINSIYET", dropna=False)["TEST_DEGERI"]
              .agg(count="count", mean="mean", std="std", min="min", median="median", max="max")).reset_index()

    # Dosya kırılımı (vektörize)
    by_file = (sub.groupby("SOURCE_FILE", dropna=False)["TEST_DEGERI"]
               .agg(count="count", mean="mean", std="std", min="min", median="median", max="max")).reset_index()

    # Test
    msg, test_info = nonparametric_test_by_group(sub, "TEST_DEGERI", "CINSIYET")

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

    # Sekmeler; grafikler isteğe bağlı
    tabs = st.tabs(["Tanımlayıcı", "Cinsiyet", "Dosya", "İstatistiksel Test", "Histogram", "Boxplot"])
    with tabs[0]:
        st.table(pd.DataFrame([stats_overall]))
    with tabs[1]:
        st.dataframe(by_sex, use_container_width=True)
    with tabs[2]:
        st.dataframe(by_file, use_container_width=True)
    with tabs[3]:
        st.info(msg)
    with tabs[4]:
        if st.checkbox(f"Histogram göster ({test_name})", value=False):
            make_hist(sub, "TEST_DEGERI", bins=30, title=f"{test_name} - Histogram")
    with tabs[5]:
        if st.checkbox(f"Boxplot göster ({test_name})", value=False):
            make_boxplot(sub, "CINSIYET", "TEST_DEGERI", title=f"{test_name} - Cinsiyete Göre Boxplot")

# Toplu özet
if results_rows:
    st.header("🧾 Toplu Özet Tablosu (Seçili Tetkikler)")
    res_df = pd.DataFrame(results_rows)
    st.dataframe(res_df, use_container_width=True)
    export_df(res_df, name="tetkik_ozet.csv")

# ================= Otomatik Rapor (Tüm Tetkikler) ================= #
st.header("📑 Otomatik Rapor (Tüm Tetkikler)")
if st.button("Raporu üret (tam veri)"):
    rows = []
    for t in sorted(df["TETKIK_ISMI"].dropna().astype(str).unique()):
        sub = df[df["TETKIK_ISMI"].astype(str) == t].copy()
        sub["TEST_DEGERI"] = pd.to_numeric(sub["TEST_DEGERI"], errors="coerce")
        sub = sub.dropna(subset=["TEST_DEGERI"])
        if sub.empty:
            continue
        stats_overall = descr_stats_fast(sub["TEST_DEGERI"])
        msg, _ = nonparametric_test_by_group(sub, "TEST_DEGERI", "CINSIYET")
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

st.caption("Performans ipuçları: Çok dosya yüklerken Polars seçeneğini açın, grafikleri ihtiyaç duydukça gösterin, büyük tabloları CSV olarak indirip lokal inceleyin.")
'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

'/mnt/data/app_optimized.py'
# ================= Klinik Tablo (β-talasemi tarzı) ================= #
st.header("📋 Klinik Tablo — Hematolojik Bulgular (Otomatik)")

src_choice = st.radio(
    "Veri kaynağı",
    ["Seçimdeki veri (work)", "Tüm veri (df)"],
    index=0,
    horizontal=True
)
base = work.copy() if src_choice.startswith("Seçimdeki") else df.copy()

# Hedef parametreler ve referans aralıkları (gerekirse düzenleyebilirsin)
params = [
    ("Age (years)", None),   # Yaş yoksa otomatik boş kalır
    ("Hb (g/dL)", "Hb"),     # (TETKIK_ISMI'ndeki ad)
    ("HCT (%)", "HCT"),
    ("RBC (×10⁶)", "RBC"),
    ("RDW (%)", "RDW"),
    ("MCV (fL)", "MCV"),
    ("MCH (pg)", "MCH"),
    ("MCHC (g/dL)", "MCHC"),
    ("HbA (%)", "HbA"),
    ("HbA₂ (%)", "HbA2"),
    ("Hb F (%)", "HbF"),
]

ref_ranges = {
    "Hb (g/dL)": "F: 11–15; M: 12–17",
    "HCT (%)":   "F: 36–46; M: 40–53",
    "RBC (×10⁶)": "F: 3.9–5.6; M: 4.5–6.0",
    "RDW (%)":   "11–16",
    "MCV (fL)":  "80–100",
    "MCH (pg)":  "27–34",
    "MCHC (g/dL)": "32–36",
    "HbA (%)":   "94–98",
    "HbA₂ (%)":  "2–3.5",
    "Hb F (%)":  "0–2",
    "Age (years)": "—",
}

# Cinsiyet normalizasyonu
sex_map = {
    "e": "Male", "erkek": "Male", "m": "Male", "male": "Male",
    "k": "Female","kadın":"Female","f":"Female","female":"Female"
}
base["__SEX__"] = (
    base["CINSIYET"].astype(str).str.strip().str.lower().map(sex_map)
)
# TEST_DEGERI sayısal güvence
base["__VAL__"] = pd.to_numeric(
    base["TEST_DEGERI"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)

def fmt_mean_sd(s: pd.Series, nd=2):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return "—"
    return f"{s.mean():.{nd}f} ± {s.std(ddof=1):.{nd}f}"

rows = []
for label, tetkik_ismi in params:
    # Age yoksa "—" bırak (datasetinde AGE sütunu varsa 'AGE'→numeric yapıp buraya entegre edebilirsin)
    if tetkik_ismi is None:
        female = "—"
        male = "—"
    else:
        sub = base[base["TETKIK_ISMI"].astype(str) == tetkik_ismi].copy()
        female = fmt_mean_sd(sub.loc[sub["__SEX__"]=="Female", "__VAL__"])
        male   = fmt_mean_sd(sub.loc[sub["__SEX__"]=="Male", "__VAL__"])

    rows.append({
        "Parameter": label,
        "Female (Mean ± SD)": female,
        "Male (Mean ± SD)": male,
        "Reference range": ref_ranges.get(label, "—")
    })

table_df = pd.DataFrame(rows)

# Görüntüle + indir
st.dataframe(table_df, use_container_width=True)
csv_bytes = table_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ Tabloyu CSV indir", data=csv_bytes, file_name="klinik_tablo.csv", mime="text/csv")

st.caption(
    "Not: Cinsiyet eşleştirmesi E/K/Erkek/Kadın/Male/Female yazımlarını otomatik algılar. "
    "Yaş (Age) veriniz yoksa satır '—' kalır. İstersen AGE/YAŞ sütununu eklersen hesaplarım."
)
