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
# Kategorik (metin) testler
# NOT: A0 HPLC bir % değerdir → KATEGORIK DEĞİL!
CATEGORICAL_TESTS = {"Kan Grubu/", "Anormal Hb/"}


# --- Erişkin pozitiflik eşikleri (TETKIK_ISMI anahtarları) ---
THRESHOLDS = {
    # HbA2
    "HbA2 (%)": (">=", 3.5),
    "A2/":      (">=", 3.5),   # sizin isimlendirme
    # HbF
    "HbF (%)":  (">",  2.0),
    "F/":       (">",  2.0),
    # Varyant yüzdeleri
    "HbS (%)":  (">",  0.0),
    "HbC (%)":  (">",  0.0),
    "HbD (%)":  (">",  0.0),
    "HbE (%)":  (">",  0.0),
}

# İsteğe bağlı: yalnızca >0 filtresini varsayılan açık yapmak istediğin test adları
GT_ZERO_DEFAULT = {
    "HbS (%)","HbC (%)","HbD (%)","HbE (%)","HbF (%)","HbA2 (%)","A2/","F/",
    "C/","D/","E/","S/"          # HPLC varyant pikleri
}


# >0 ise "pozitif" sayılacak varyant yüzdeleri (ihtiyacına göre genişlet)
VARIANT_NUMERIC_TESTS = {
    "HbS (%)", "HbC (%)", "HbD (%)", "HbE (%)",
    "HbF (%)", "HbA2 (%)",   # eşik kullanacaksan ayrıca ekleriz
    "Anormal Hb/"            # sayı geliyorsa >0 filtre uygular
}
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
    
# __VAL_NUM__ kolonunu güvenle oluşturur (yoksa ekler)
def add_numeric_copy(frame, src_col="TEST_DEGERI", out_col="__VAL_NUM__"):
    if out_col not in frame.columns:
        tmp = (
            frame[src_col].astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
        frame[out_col] = pd.to_numeric(tmp, errors="coerce")
    return frame
    
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
        
def add_numeric_copy(frame, src_col="TEST_DEGERI", out_col="__VAL_NUM__"):
    if out_col not in frame.columns:
        tmp = (frame[src_col].astype(str)
               .str.replace(",", ".", regex=False)
               .str.replace(" ", "", regex=False))
        frame[out_col] = pd.to_numeric(tmp, errors="coerce")
    return frame

def apply_threshold(series, rule):
    op, cut = rule
    if op == ">=": return series >= cut
    if op == ">":  return series >  cut
    if op == "<=": return series <= cut
    if op == "<":  return series <  cut
    return series.notna()
    
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

# ================= VARYANT ÖZETLERİ (Anormal Hb / HPLC → Etiket + Özet) ================= #
# === VARYANT ÖZETİ / ETİKETLEME === (BAŞLA)
import re

# Güvence: numeric kopya olsun
work = add_numeric_copy(work)

# Erişkin eşik setleri
A2_KEYS = {"A2/", "HbA2", "HbA2 (%)", "Hb A2", "Hb A2 (%)"}
F_KEYS = {"F/", "HbF", "HbF (%)", "Hb F", "Hb F (%)"}

# TETKIK_ISMI -> varyant adı (HPLC pikleri)
NUMVAR_FROM_TEST = {
    "C/": "HbC",
    "D/": "HbD",
    "E/": "HbE",
    "S/": "HbS",
}


def norm_anormal_hb_text(x: str | None):
    if not isinstance(x, str):
        return None
    s = x.upper().replace("İ", "I").strip()
    if re.search(r"S-?BETA|S ?β", s):
        return "Hb S-β-thal"
    if re.search(r"\bHBS\b|S TRAIT|S HET|HBS HET|HBS TAS|S-TASIY", s):
        return "HbS"
    if re.search(r"\bHBC\b", s):
        return "HbC"
    if re.search(r"\bHBD\b", s):
        return "HbD"
    if re.search(r"\bHBE\b", s):
        return "HbE"
    if re.search(r"\bA2\b|HBA2", s):
        return "HbA2↑"
    if re.search(r"\bF\b|HBF", s):
        return "HbF↑"
    if re.search(r"\bNORMAL\b|NEG", s):
        return "Normal"
    return None


def pick_variant_tag(g: pd.DataFrame) -> str | None:
    g = add_numeric_copy(g.copy())
    g["TETKIK_ISMI"] = g["TETKIK_ISMI"].astype(str)
    tags = []

    # 1) Anormal Hb/ metinlerinden
    txt = g.loc[g["TETKIK_ISMI"] == "Anormal Hb/", "TEST_DEGERI"].dropna().astype(str)
    for v in txt:
        t = norm_anormal_hb_text(v)
        if t:
            tags.append(t)

    # 2) HbA2 ve HbF erişkin eşikleri
    if g["TETKIK_ISMI"].isin(A2_KEYS).any():
        a2 = g.loc[g["TETKIK_ISMI"].isin(A2_KEYS), "__VAL_NUM__"].dropna()
        if not a2.empty and a2.max() >= 3.5:
            tags.append("HbA2↑")
    if g["TETKIK_ISMI"].isin(F_KEYS).any():
        f = g.loc[g["TETKIK_ISMI"].isin(F_KEYS), "__VAL_NUM__"].dropna()
        if not f.empty and f.max() > 2.0:
            tags.append("HbF↑")

    # 3) HPLC pikleri (C/D/E/S) -> >0 varsa ilgili varyant
    for k, var_name in NUMVAR_FROM_TEST.items():
        m = g["TETKIK_ISMI"] == k
        if m.any():
            vv = g.loc[m, "__VAL_NUM__"].dropna()
            if not vv.empty and (vv > 0).any():
                tags.append(var_name)

    if not tags:
        return None
    priority = ["Hb S-β-thal", "HbS", "HbC", "HbD", "HbE", "HbA2↑", "HbF↑", "Normal"]
    for p in priority:
        if p in tags:
            return p
    return tags[0]
# === VARYANT ÖZETİ / ETİKETLEME === (BİTİR)


if "VARIANT_TAG" not in work.columns:
    # __VAL_NUM__ güvence (tüm work üzerinde)
    work = add_numeric_copy(work)

    var_map = (work.groupby("PROTOKOL_NO", group_keys=False)
                  .apply(lambda g: pd.Series({"VARIANT_TAG": pick_variant_tag(g)}))
                  .reset_index())
    work = work.merge(var_map, on="PROTOKOL_NO", how="left")


# Kullanıcı arayüzü
st.header("📋 Varyant Özeti — erişkin eşikleri ile")
st.caption(
    "Son halini Streamlit arayüzünde anlık görebilmek için `streamlit run app.py` komutunu çalıştırabilirsiniz. "
    "Bu bölüm, yüklediğiniz verilere göre her yeniden çalıştırmada güncellenir."
)

# Sadece anlamlı etiketleri (harfler) göster
order = ["Hb S-β-thal", "HbS", "HbC", "HbD", "HbE", "HbA2↑", "HbF↑", "Normal"]
present = [t for t in order if t in set(work["VARIANT_TAG"].dropna())]

variant_choice = st.selectbox("Varyant seç:", ["(Tümü)"] + present, index=0)

base_v = work.copy()
if variant_choice != "(Tümü)":
    base_v = base_v[base_v["VARIANT_TAG"] == variant_choice]

# 1) (Tümü) ise: frekans tablosu
if variant_choice == "(Tümü)":
    freq = (work["VARIANT_TAG"]
            .value_counts(dropna=True)
            .rename_axis("Varyant").to_frame("N").reset_index())
    total = int(freq["N"].sum()) if not freq.empty else 0
    if total > 0:
        freq["%"] = (freq["N"] / total * 100).round(2)
    st.subheader("Varyant Frekansları")
    st.dataframe(freq, use_container_width=True)
    st.download_button("⬇️ Varyant frekansları (CSV)",
                       data=freq.to_csv(index=False).encode("utf-8-sig"),
                       file_name="varyant_frekans.csv", mime="text/csv")

# 2) Seçilen varyant için Female/Male Mean±SD tablosu ve olgu listesi
def _mean_sd(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return "—" if s.empty else f"{s.mean():.2f} ± {s.std(ddof=1):.2f}"

PARAMS = {
    "Hemogram/HGB":  ("Hb (g/dL)",   "F: 11–15; M: 12–17"),
    "Hemogram/HCT":  ("HCT (%)",     "F: 36–46; M: 40–53"),
    "Hemogram/RBC":  ("RBC (×10⁶)",  "F: 3.9–5.6; M: 4.5–6.0"),
    "Hemogram/RDW":  ("RDW (%)",     "11–16"),
    "Hemogram/MCV":  ("MCV (fL)",    "80–100"),
    "Hemogram/MCH":  ("MCH (pg)",    "27–34"),
    "Hemogram/MCHC": ("MCHC (g/dL)", "32–36"),
    "HbA":           ("HbA (%)",     "94–98"),
    "HbA2 (%)":      ("HbA₂ (%)",    "2–3.5"),
    "A2/":           ("HbA₂ (%)",    "2–3.5"),
    "HbF (%)":       ("Hb F (%)",    "0–2"),
    "F/":            ("Hb F (%)",    "0–2"),
}

st.subheader("♀/♂ Mean ± SD (seçilen varyant)")
rows = []
for tetkik_key, (disp, ref) in PARAMS.items():
    subp = base_v[base_v["TETKIK_ISMI"] == tetkik_key].copy()
    if subp.empty: 
        continue
    subp = add_numeric_copy(subp)  # __VAL_NUM__ güvence
    fem = _mean_sd(subp.loc[subp["CINSIYET"].astype(str).str.lower().str.startswith(("k","f")), "__VAL_NUM__"])
    male= _mean_sd(subp.loc[subp["CINSIYET"].astype(str).str.lower().str.startswith(("e","m")), "__VAL_NUM__"])
    rows.append({"Parameter": disp, "Female (Mean ± SD)": fem, "Male (Mean ± SD)": male, "Reference range": ref})

table_fm = pd.DataFrame(rows)
if table_fm.empty and variant_choice != "(Tümü)":
    st.info("Bu varyant için parametrik veri bulunamadı.")
else:
    st.dataframe(table_fm, use_container_width=True)
    st.download_button("⬇️ Tablo #1 (CSV)",
                       data=table_fm.to_csv(index=False).encode("utf-8-sig"),
                       file_name="varyant_ozet_female_male.csv", mime="text/csv")

# Olgu listesi: seçili varyantta kimler var?
if variant_choice != "(Tümü)":
    cols_keep = ["PROTOKOL_NO", "TCKIMLIK_NO", "CINSIYET", "SOURCE_FILE"]
    cols_keep = [c for c in cols_keep if c in base_v.columns]
    case_tbl = base_v.drop_duplicates(subset=["PROTOKOL_NO"])[cols_keep + ["VARIANT_TAG"]]
    st.subheader("Olgu listesi (seçilen varyant)")
    st.dataframe(case_tbl, use_container_width=True)
    st.download_button("⬇️ Olgu listesi (CSV)",
                       data=case_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"olgu_listesi_{variant_choice}.csv", mime="text/csv")


# ================= Ön-izleme & Müdahale: Metinden Sayıya ================= #
import re
import numpy as np
import pandas as pd
import streamlit as st

# --- Yardımcılar (esnek sayı çözücü) ---
def _dec_fix(x: str) -> str:
    s = x.replace("\xa0", " ").strip()
    # hem '.' hem ',' varsa: sondaki ayraç ondalık, diğerleri binlik sayılır
    if "," in s and "." in s:
        last = max(s.rfind(","), s.rfind("."))
        dec = s[last]
        s = re.sub(r"[.,](?=\d{3}\b)", "", s)  # binlikleri at
        s = s.replace(dec, ".")
    elif "," in s:
        s = s.replace(".", "")         # olası binlik noktaları
        s = s.replace(",", ".")        # ondalık virgül → nokta
    else:
        s = re.sub(r"\.(?=\d{3}\b)", "", s)  # binlik noktayı at
    return s

def smart_number(text: str):
    """Öneri üretir (float veya None). Birimleri, %, < >, aralıkları yakalar."""
    if text is None: 
        return None
    s = str(text).strip().lower()
    if s in {"", "nan", "na", "n/a", "yok", "boş", "empty", "nd"}:
        return None
    if s in {"pozitif", "+", "positive", "pos"}: return 1.0
    if s in {"negatif", "-", "negative", "neg"}: return 0.0

    # yüzde (15% → 15)
    s = s.replace("%", " ")

    # eşitsizlikler (<, ≤, >, ≥) → sınır değer (isteğe göre geliştirebiliriz)
    m_ineq = re.match(r"^\s*([<>]=?)\s*([0-9.,]+)", s)
    if m_ineq:
        op, num = m_ineq.groups()
        try: v = float(_dec_fix(num))
        except: v = None
        return v

    # aralıklar (12–14, 12-14) → ortalama
    m_rng = re.match(r"^\s*([+-]?\d[\d.,]*)\s*[-–—]\s*([+-]?\d[\d.,]*)", s)
    if m_rng:
        a, b = m_rng.groups()
        try:
            a = float(_dec_fix(a)); b = float(_dec_fix(b))
            return (a + b) / 2.0
        except:
            return None

    # metin içindeki ilk sayıyı çek (12,3 g/dL vb.)
    m_any = re.search(r"[+-]?\d[\d.,]*", s)
    if m_any:
        try: return float(_dec_fix(m_any.group(0)))
        except: return None

    return None

# --- Problem yakalama: Öneri + işaretleme ---
orig = work["TEST_DEGERI"].astype(str)
suggested = orig.map(smart_number)

# “problemli” kriteri: numeric'e çevrilemiyor ya da metinde işaret/harf/aralık var
is_categorical_row = work["TETKIK_ISMI"].astype(str).isin(CATEGORICAL_TESTS)
mask_problem = (
    (~is_categorical_row) & (
        suggested.isna() |
        orig.str.contains(r"[A-Za-z%<>]|[-–—].*[-–—]", regex=True)
    )
)

# İnceleme tablosu: orijinal + öneri + düzenlenebilir hedef
preview = work.loc[mask_problem, [
    "TETKIK_ISMI", "CINSIYET", "PROTOKOL_NO", "TCKIMLIK_NO", "TEST_DEGERI"
]].copy()

preview = preview.reset_index().rename(columns={"index": "__ROW_ID__"})
preview["SUGGESTED"] = suggested.loc[preview["__ROW_ID__"]].values

st.header("🧹 Problemli Değerler — Ön-izleme & Müdahale")
st.caption("Aşağıda metin içerikli/sorunlu tüm değerler listelenir. 'CLEAN_VALUE' sütununu elle düzeltebilirsin.")

# Düzenlenebilir sütun: CLEAN_VALUE (başlangıçta öneri)
preview["CLEAN_VALUE"] = preview["SUGGESTED"]

edited = st.data_editor(
    preview,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "__ROW_ID__": st.column_config.NumberColumn(label="RowID", help="Orijinal satır indeksi", disabled=True),
        "TEST_DEGERI": st.column_config.TextColumn(label="ORIGINAL", help="Ham değer", disabled=True),
        "SUGGESTED": st.column_config.NumberColumn(label="ÖNERİ (otomatik)", help="Algoritmanın önerdiği", disabled=True),
        "CLEAN_VALUE": st.column_config.NumberColumn(label="CLEAN_VALUE (elle düzenle)", help="Burayı istediğin gibi değiştir"),
    },
    hide_index=True
)

col_apply, col_opts = st.columns([1,1])
with col_apply:
    apply_now = st.button("✅ Düzenlemeleri uygula (TEST_DEGERI_CLEAN)")

with col_opts:
    overwrite_main = st.checkbox("TEST_DEGERI sütununu CLEAN_VALUE ile değiştir", value=False)

if apply_now:
    # Son kullanıcının girdiklerini orijinal work'e geri yaz
    updates = edited[["__ROW_ID__", "CLEAN_VALUE"]].dropna(subset=["__ROW_ID__"])
    work.loc[updates["__ROW_ID__"].values, "TEST_DEGERI_CLEAN"] = updates["CLEAN_VALUE"].values

    if overwrite_main:
        work.loc[updates["__ROW_ID__"].values, "TEST_DEGERI"] = updates["CLEAN_VALUE"].values

    st.success(
        f"Güncellendi: {updates['__ROW_ID__'].nunique():,} satır için CLEAN_VALUE uygulandı. "
        f"{'TEST_DEGERI de güncellendi.' if overwrite_main else 'TEST_DEGERI_CLEAN sütunu oluşturuldu/güncellendi.'}"
    )
    st.download_button(
        "⬇️ Temiz/duzeltilmiş veriyi indir (CSV)",
        data=work.to_csv(index=False).encode("utf-8-sig"),
        file_name="temizlenmis_veri.csv",
        mime="text/csv"
    )



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

    # --- KATEGORİK TESTLER (Kan Grubu, Anormal Hb) ---
    if test_name in {"Kan Grubu/", "Anormal Hb/"}:
        st.info("Bu tetkik kategorik olarak değerlendirildi (frekans analizi).")

        import re
        def normalize_blood_group(x):
            if not isinstance(x, str): return None
            s = x.upper().replace("İ","I").strip()
            abo = None
            if re.search(r"\bAB\b", s): abo = "AB"
            elif re.search(r"\bA\b", s): abo = "A"
            elif re.search(r"\bB\b", s): abo = "B"
            elif re.search(r"\b0\b|\bO\b", s): abo = "O"
            rh = "Rh(+)" if re.search(r"(\+|POS|POZ|RH\+)", s) else ("Rh(-)" if re.search(r"(\-|NEG)", s) else "")
            return (abo + (" " + rh if rh else "")).strip() if abo else s

        def normalize_anormal_hb_text(x):
            if not isinstance(x, str): return None
            s = x.upper().replace("İ","I").strip()
            if re.search(r"S-?BETA|S ?β", s): return "Hb S-β-thal"
            if re.search(r"\bHBS\b|S TRAIT|S HET|HBS HET|HBS TAS|S-TASIY", s): return "HbS"
            if re.search(r"\bHBC\b", s): return "HbC"
            if re.search(r"\bHBD\b", s): return "HbD"
            if re.search(r"\bHBE\b", s): return "HbE"
            if re.search(r"\bA2\b|HBA2", s): return "HbA2↑"
            if re.search(r"\bF\b|HBF", s): return "HbF↑"
            if re.search(r"\bNORMAL\b|NEG", s): return "Normal"
            return s or None

        if test_name == "Kan Grubu/":
            cat_series = sub["TEST_DEGERI"].map(normalize_blood_group)
        else:
            cat_series = sub["TEST_DEGERI"].map(normalize_anormal_hb_text)

        sub_cat = sub.assign(__CAT__=cat_series)

        # ---- Frekans tablosu ----
        freq_all = (sub_cat["__CAT__"].value_counts(dropna=False)
                    .rename_axis("Kategori").to_frame("N").reset_index())
        total = int(freq_all["N"].sum()) if not freq_all.empty else 0
        if total > 0:
            freq_all["%"] = (freq_all["N"]/total*100).round(2)

        freq_by_sex = (sub_cat.pivot_table(index="__CAT__", columns="CINSIYET",
                                           values="PROTOKOL_NO", aggfunc="count", fill_value=0)
                       .astype(int).reset_index().rename(columns={"__CAT__":"Kategori"}))

        from scipy.stats import chi2_contingency
        chi2_msg = "Ki-kare uygulanamadı."
        try:
            cont = freq_by_sex.drop(columns=["Kategori"]).values
            if cont.sum() > 0 and cont.shape[1] > 1:
                chi2, p, dof, _ = chi2_contingency(cont)
                chi2_msg = f"Chi-square: χ²={chi2:.2f}, df={dof}, p={p:.4g}"
        except Exception as e:
            chi2_msg = f"Hata: {e}"

        tabs = st.tabs(["Frekans", "Cinsiyet Dağılımı", "İstatistik", "Olgu Listesi"])
        with tabs[0]:
            st.dataframe(freq_all, use_container_width=True)
        with tabs[1]:
            st.dataframe(freq_by_sex, use_container_width=True)
        with tabs[2]:
            st.info(chi2_msg)
        with tabs[3]:
            keep = ["PROTOKOL_NO","TCKIMLIK_NO","CINSIYET","SOURCE_FILE"]
            keep = [c for c in keep if c in sub_cat.columns]
            case_tbl = sub_cat[keep + ["TEST_DEGERI","__CAT__"]].rename(columns={"__CAT__":"Kategori"})
            st.dataframe(case_tbl, use_container_width=True)
            st.download_button("⬇️ Olgu listesi (CSV)",
                data=case_tbl.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{test_name}_olgu_listesi.csv", mime="text/csv")

        results_rows.append({
            "TETKIK_ISMI": test_name,
            "N": int(freq_all["N"].sum()),
            "Mean": None, "Median": None, "Std": None,
            "Min": None, "Q1": None, "Q3": None, "Max": None,
            "Normalite": "—", "Test": chi2_msg
        })


    # sayısal kopya hazırla
    sub = add_numeric_copy(sub)

    st.subheader(f"🧷 {test_name}")

    # --- KATEGORİK Mİ? ---
    is_categorical = (test_name in CATEGORICAL_TESTS)
    if not is_categorical:
        # veri temelli kontrol (sayısal oranı çok düşükse kategorik say)
        num_ratio = sub["__VAL_NUM__"].notna().mean()
        if num_ratio < 0.3:
            is_categorical = True

    if is_categorical:
        st.info("Bu tetkik kategorik olarak değerlendirildi (frekans analizi).")

        freq_all = (sub["TEST_DEGERI"].astype(str).str.strip()
                    .value_counts(dropna=False).rename_axis("Kategori")
                    .to_frame("N").reset_index())
        freq_all["%"] = (freq_all["N"] / freq_all["N"].sum() * 100).round(2)

        freq_by_sex = (sub.pivot_table(index="TEST_DEGERI", columns="CINSIYET",
                                       values="PROTOKOL_NO", aggfunc="count", fill_value=0)
                       .astype(int).reset_index().rename(columns={"TEST_DEGERI":"Kategori"}))

        chi2_msg = "Ki-kare uygulanamadı."
        try:
            from scipy.stats import chi2_contingency
            cont = freq_by_sex.drop(columns=["Kategori"]).values
            if cont.sum() > 0 and cont.shape[1] > 1:
                chi2, p, dof, _ = chi2_contingency(cont)
                chi2_msg = f"Chi-square: χ²={chi2:.2f}, df={dof}, p={p:.4g}"
        except Exception as e:
            chi2_msg = f"Hata: {e}"

        tabs = st.tabs(["Frekans", "Cinsiyet Dağılımı", "İstatistik"])
        with tabs[0]: st.dataframe(freq_all, use_container_width=True)
        with tabs[1]: st.dataframe(freq_by_sex, use_container_width=True)
        with tabs[2]: st.info(chi2_msg)

        results_rows.append({
            "TETKIK_ISMI": test_name, "N": int(freq_all["N"].sum()),
            "Mean": None, "Median": None, "Std": None, "Min": None, "Q1": None, "Q3": None, "Max": None,
            "Normalite": "—", "Test": chi2_msg
        })
        continue  # sayısal kola girme

    # --- SAYISAL ANALİZ (tek çerçeve: sub_work) ---
    use_threshold = st.checkbox(
        f"‘{test_name}’ için erişkin eşiğini uygula",
        value=(test_name in THRESHOLDS),
        key=f"th_{test_name}"
    )
    use_gt_zero  = st.checkbox(
        f"‘{test_name}’ için sadece > 0 değerleri dahil et",
        value=(test_name in GT_ZERO_DEFAULT),
        key=f"gt0_{test_name}"
    )

    sub_work = sub[sub["__VAL_NUM__"].notna()].copy()
    if use_threshold and test_name in THRESHOLDS:
        sub_work = sub_work[apply_threshold(sub_work["__VAL_NUM__"], THRESHOLDS[test_name])]
        st.caption(f"Eşik: {THRESHOLDS[test_name][0]} {THRESHOLDS[test_name][1]}")
    elif use_gt_zero:
        sub_work = sub_work[sub_work["__VAL_NUM__"] > 0]
        st.caption("Filtre: > 0")

    if sub_work.empty:
        st.warning("Filtre sonrası satır bulunamadı.")
        continue

    # Tanımlayıcılar
    stats_overall = descr_stats_fast(sub_work["__VAL_NUM__"])
    normal_flag   = normality_flag(sub_work["__VAL_NUM__"])

    # Cinsiyet / Dosya kırılımları
    by_sex = (sub_work.groupby("CINSIYET", dropna=False)["__VAL_NUM__"]
              .agg(count="count", mean="mean", std="std", min="min", median="median", max="max")).reset_index()

    by_file = (sub_work.groupby("SOURCE_FILE", dropna=False)["__VAL_NUM__"]
               .agg(count="count", mean="mean", std="std", min="min", median="median", max="max")).reset_index()

    # Nonparametrik test
    _msg_df = sub_work.rename(columns={"__VAL_NUM__": "VAL"})
    msg, test_info = nonparametric_test_by_group(_msg_df, "VAL", "CINSIYET")

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

    # Sekmeler
    tabs = st.tabs(["Tanımlayıcı", "Cinsiyet", "Dosya", "İstatistiksel Test", "Histogram", "Boxplot"])
    with tabs[0]: st.table(pd.DataFrame([stats_overall]))
    with tabs[1]: st.dataframe(by_sex, use_container_width=True)
    with tabs[2]: st.dataframe(by_file, use_container_width=True)
    with tabs[3]: st.info(msg)
    with tabs[4]:
        if st.checkbox(f"Histogram göster ({test_name})", value=False):
            make_hist(_msg_df, "VAL", bins=30, title=f"{test_name} - Histogram")
    with tabs[5]:
        if st.checkbox(f"Boxplot göster ({test_name})", value=False):
            make_boxplot(sub_work, "CINSIYET", "__VAL_NUM__", title=f"{test_name} - Cinsiyete Göre Boxplot")

    # Pozitif (filtre sonrası) liste
    pos_cols = ["PROTOKOL_NO", "TCKIMLIK_NO", "CINSIYET", "SOURCE_FILE"]
    pos_cols = [c for c in pos_cols if c in sub_work.columns]
    pos_tbl = sub_work[pos_cols + ["__VAL_NUM__"]].sort_values("__VAL_NUM__", ascending=False)
    st.write("**Filtre sonrası kayıtlar**")
    st.dataframe(pos_tbl, use_container_width=True)
    st.download_button(
        "⬇️ TCKIMLIK_NO listesi (CSV)",
        data=pos_tbl.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{test_name}_filtre_sonrasi.csv",
        mime="text/csv"
    )


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
