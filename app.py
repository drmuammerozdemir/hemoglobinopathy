# app.py
# -*- coding: utf-8 -*-
"""
🧪 Tetkik Analiz Arayüzü — Çoklu Dosya (Optimize, Revize)
- Çoklu dosya hızlı okuma (paralel + cache)
- Bellek optimizasyonu (downcast, categorical)
- İsteğe bağlı Polars hızlandırma
- Büyük tabloları güvenli göstermeye yönelik limitler
- Grafikler isteğe bağlı (matplotlib; renk set edilmez)
- Kategorik analizlerde SAĞLAM normalizasyon:
    • Kan Grubu: A/B/AB/O/0 + Rh(+/-/poz/neg/rh+/rh-) → tek tipe
    • Anormal Hb: HbS/HbC/HbD/HbE/HbA2↑/HbF↑/Normal
- Hem ham yazımlar hem normalize edilmiş kategoriler ayrı tablolar/CSV
- Ham yazımdan hasta/protokol seçerek hastanın/protokolün tüm tetkiklerini göster

Çalıştırma:
    streamlit run app.py
"""

import io
import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ThreadPoolExecutor

# ============== Ayarlar ============== #
st.set_page_config(page_title="Tetkik Analiz — Optimize", layout="wide")

REQ_COLS = ["PROTOKOL_NO", "TCKIMLIK_NO", "TETKIK_ISMI", "TEST_DEGERI", "CINSIYET"]

# Kategorik (metin) testler
CATEGORICAL_TESTS = {"Kan Grubu/", "Anormal Hb/"}

# --- Erişkin pozitiflik eşikleri (TETKIK_ISMI anahtarları) ---
THRESHOLDS = {
    "HbA2 (%)": (">=", 3.5),
    "A2/":      (">=", 3.5),
    "HbF (%)":  (">",  2.0),
    "F/":       (">",  2.0),
    "HbS (%)":  (">",  0.0),
    "HbC (%)":  (">",  0.0),
    "HbD (%)":  (">",  0.0),
    "HbE (%)":  (">",  0.0),
}
GT_ZERO_DEFAULT = {
    "HbS (%)","HbC (%)","HbD (%)","HbE (%)","HbF (%)","HbA2 (%)","A2/","F/",
    "C/","D/","E/","S/"
}
VARIANT_NUMERIC_TESTS = {
    "HbS (%)","HbC (%)","HbD (%)","HbE (%)","HbF (%)","HbA2 (%)","Anormal Hb/"
}

DISPLAY_LIMIT = 400

MALE_TOKENS   = {"e","erkek","m","male","bay"}
FEMALE_TOKENS = {"k","kadın","kadin","f","female","bayan"}

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

# ----- P değeri yazım kuralı (Türkçe ondalık) -----
def _fmt_p(p: float) -> str:
    if p is None or np.isnan(p):
        return "—"
    if p < 0.001:
        return "<0,001"
    if p < 0.05:
        return "<0,05"
    return f"{p:.3f}".replace(".", ",")

# ----- Normalite testi: n<=5000 Shapiro; büyük n KS (N(μ,σ)) -----
def normality_test_with_p(series: pd.Series, alpha: float = 0.05):
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n < 3:
        return "yetersiz", "—"

    try:
        if n <= 5000:
            stat, p = stats.shapiro(x)
        else:
            mu = float(np.mean(x))
            sd = float(np.std(x, ddof=1))
            if sd == 0:
                return "yetersiz", "—"
            # H0: veri ~ N(mu, sd)
            stat, p = stats.kstest(x, 'norm', args=(mu, sd))

        label = "normal" if p >= alpha else "non-normal"
        return label, _fmt_p(p)
    except Exception:
        return "bilinmiyor", "—"

def add_numeric_copy(frame, src_col="TEST_DEGERI", out_col="__VAL_NUM__"):
    if out_col not in frame.columns:
        tmp = (frame[src_col].astype(str)
               .str.replace(",", ".", regex=False)
               .str.replace(" ", "", regex=False))
        frame[out_col] = pd.to_numeric(tmp, errors="coerce")
    return frame


def check_columns(df: pd.DataFrame):
    return [c for c in REQ_COLS if c not in df.columns]


def normalize_sex_label(value):
    if not isinstance(value, str): return None
    trimmed = value.strip()
    if not trimmed: return None
    low = trimmed.lower()
    if low in MALE_TOKENS: return "Erkek"
    if low in FEMALE_TOKENS: return "Kadın"
    return trimmed


def _resolve_patient_sex(series: pd.Series) -> str:
    values = [v for v in pd.unique(series.dropna()) if isinstance(v, str) and v]
    if not values: return "Bilinmiyor"
    if len(values) == 1: return values[0]
    return "Çakışma"


def summarize_sex_counts(frame: pd.DataFrame) -> pd.DataFrame:
    tmp = frame[["TCKIMLIK_NO", "CINSIYET"]].copy()
    tmp["CINSIYET"] = tmp["CINSIYET"].astype(str)
    tmp["__SEX_CANON__"] = tmp["CINSIYET"].map(normalize_sex_label).astype(object)
    s_rows = tmp["__SEX_CANON__"].where(tmp["__SEX_CANON__"].notna(), "Bilinmiyor")
    row_counts = (
        s_rows.value_counts(dropna=False)
        .rename_axis("CINSIYET").to_frame("Satır Sayısı")
    )
    with_id = tmp[tmp["TCKIMLIK_NO"].notna()].copy()
    if not with_id.empty:
        w = with_id.copy()
        w["__SEX_CANON__"] = w["__SEX_CANON__"].astype(object)
        patient_gender = (
            w.groupby("TCKIMLIK_NO")["__SEX_CANON__"]
             .apply(lambda s: _resolve_patient_sex(pd.Series(pd.unique(s.dropna()))))
             .reset_index(name="__SEX_RESOLVED__")
        )
        patient_counts = (
            patient_gender["__SEX_RESOLVED__"].fillna("Bilinmiyor")
            .value_counts(dropna=False)
            .rename_axis("CINSIYET").to_frame("Hasta (Benzersiz)")
        )
    else:
        patient_counts = pd.DataFrame(columns=["Hasta (Benzersiz)"])
    summary = row_counts.join(patient_counts, how="outer").fillna(0)
    summary["Satır Sayısı"] = summary["Satır Sayısı"].astype(int)
    if "Hasta (Benzersiz)" in summary.columns:
        summary["Hasta (Benzersiz)"] = summary["Hasta (Benzersiz)"].astype(int)
    else:
        summary["Hasta (Benzersiz)"] = 0
    total_rows = int(summary["Satır Sayısı"].sum())
    total_patients = int(summary["Hasta (Benzersiz)"].sum())
    summary["% Satır"]  = (summary["Satır Sayısı"] / total_rows * 100).round(2) if total_rows else np.nan
    summary["% Hasta"] = (summary["Hasta (Benzersiz)"] / total_patients * 100).round(2) if total_patients else np.nan
    summary = summary.reset_index()
    summary = summary[["CINSIYET","Hasta (Benzersiz)","% Hasta","Satır Sayısı","% Satır"]]
    return summary.sort_values("Hasta (Benzersiz)", ascending=False).reset_index(drop=True)


def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    if "TEST_DEGERI" in df.columns:
        df["TEST_DEGERI"] = df["TEST_DEGERI"].astype(str)
    for col in ["TETKIK_ISMI", "CINSIYET", "SOURCE_FILE"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def descr_stats_fast(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce")
    x = x[~x.isna()]
    if x.empty:
        return {"count":0,"mean":np.nan,"std":np.nan,"min":np.nan,"q1":np.nan,"median":np.nan,"q3":np.nan,"max":np.nan,"cv%":np.nan,"iqr":np.nan}
    q = np.percentile(x, [25, 50, 75])
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    cv  = (std/mean)*100 if mean!=0 else np.nan
    return {"count":int(x.size),"mean":mean,"std":std,"min":float(x.min()),"q1":float(q[0]),"median":float(q[1]),"q3":float(q[2]),"max":float(x.max()),"cv%":float(cv),"iqr":float(q[2]-q[0])}


def normality_flag(x: pd.Series, alpha=0.05) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 3: return "yetersiz"
    try:
        if len(x) <= 5000:
            stat, p = stats.shapiro(x)
            return "normal" if p >= alpha else "non-normal"
        else:
            res = stats.anderson(x, dist="norm")
            crit = res.critical_values[2]
            return "normal" if res.statistic < crit else "non-normal"
    except Exception:
        return "bilinmiyor"


def apply_threshold(series, rule):
    op, cut = rule
    if op == ">=": return series >= cut
    if op == ">":  return series >  cut
    if op == "<=": return series <= cut
    if op == "<":  return series <  cut
    return series.notna()


def nonparametric_test_by_group(df, val_col, grp_col):
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
        st.info("Grafik için yeterli veri yok."); return
    cats = list(valid[x_col].astype(str).unique())
    data = [valid[valid[x_col].astype(str) == c][y_col].values for c in cats]
    fig, ax = plt.subplots()
    ax.boxplot(data, labels=cats, showmeans=True)
    ax.set_title(title); ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    st.pyplot(fig)


def make_hist(df, col, bins=30, title="Histogram"):
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if x.empty:
        st.info("Histogram için yeterli veri yok."); return
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins)
    ax.set_title(title); ax.set_xlabel(col); ax.set_ylabel("Frekans")
    st.pyplot(fig)


def export_df(df, name="export.csv"):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ CSV indir", data=csv, file_name=name, mime="text/csv")


# ======== ÖZEL: Kategorik normalizasyon fonksiyonları ======== #
def normalize_blood_group(x: str | None):
    """
    'A Rh (+) Pozitif' -> 'A Rh(+)', 'O Rh -' -> 'O Rh(-)', '0 +' -> 'O Rh(+)'
    metin anlaşılmazsa None döner.
    """
    if not isinstance(x, str): return None
    u = x.strip().upper().replace("İ", "I")
    if not u: return None

    # ABO (AB, A, B, O/0)
    abo = None
    if re.search(r"\bAB\b", u):
        abo = "AB"
    elif re.search(r"\bA\b", u):
        abo = "A"
    elif re.search(r"\bB\b", u):
        abo = "B"
    elif re.search(r"\bO\b|\b0\b", u):
        abo = "O"

    # Rh (+ / - / POS/POZ / NEG / RH+ / RH- / + / -)
    rh = None
    if re.search(r"\+|\bPOS(ITIVE)?\b|\bPOZ(ITIF)?\b|\bRH\+\b", u):
        rh = "Rh(+)"
    elif re.search(r"-|\bNEG(ATIVE)?\b|\bRH-\b", u):
        rh = "Rh(-)"

    if abo is None and rh is None:
        return None
    return f"{abo or ''} {rh or ''}".strip()


def norm_anormal_hb_text(x: str | None):
    if not isinstance(x, str): return None
    s = x.upper().replace("İ","I").strip()
    
    # YENİ EKLENEN KONTROL (Genellikle USV veya benzeri metinleri yakalar)
    if re.search(r"\bUSV\b|UNIDENTIFIED|TANIMLANAMAYAN", s): return "USV"
    
    # --- MEVCUT KOD ---
    if re.search(r"S-?BETA|S ?β", s): return "Hb S-β-thal"
    if re.search(r"\bHBS\b|S TRAIT|S HET|HBS HET|HBS TAS|S-TASIY", s): return "HbS"
    if re.search(r"\bHBC\b", s): return "HbC"
    if re.search(r"\bHBD\b", s): return "HbD"
    if re.search(r"\bHBE\b", s): return "HbE"
    if re.search(r"\bA2\b|HBA2", s): return "HbA2↑"
    if re.search(r"\bF\b|HBF", s): return "HbF↑"
    if re.search(r"\bNORMAL\b|NEG", s): return "Normal"
    return None


# ============== Cache'li Dosya Okuma ============== #
@st.cache_data(show_spinner=False)
def read_one_excel_cached(file_bytes: bytes, engine_hint: str = "openpyxl") -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    return pd.read_excel(bio, engine=engine_hint)


def read_many_excels(files):
    def _read(upl):
        try:
            data = upl.read()
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
st.title("⚡ Tetkik Analiz — Çoklu Dosya (Optimize, Revize)")
st.caption("Büyük veri ve çoklu dosya için hızlandırılmış sürüm (kan grubu/anormal Hb normalizasyonu dâhil).")

uploads = st.file_uploader("Excel dosyaları (.xlsx, .xls) — Çoklu seçim", type=["xlsx", "xls"], accept_multiple_files=True)

use_polars = st.checkbox("Polars hızlandırmayı dene (kuruluysa)", value=('pl' in globals() and HAS_POLARS),
                         help="Polars kurulu değilse otomatik devre dışı kalır.")

if not uploads:
    st.info("Birden çok dosyayı aynı anda seçin (örn. 12 dosya).")
    st.stop()

with st.spinner("Dosyalar okunuyor..."):
    results = read_many_excels(uploads)

frames, skipped = [], []
for name, tmp, err in results:
    if err:
        skipped.append((name, f"Okuma hatası: {err}")); continue
    miss = check_columns(tmp)
    if miss:
        skipped.append((name, f"Eksik sütun: {miss}")); continue
    tmp["SOURCE_FILE"] = name
    frames.append(tmp)

if skipped:
    for nm, msg in skipped:
        st.warning(f"'{nm}' atlandı → {msg}")

if not frames:
    st.error("Uygun veri içeren dosya bulunamadı."); st.stop()

df = pd.concat(frames, ignore_index=True)
df = downcast_df(df)

if use_polars and HAS_POLARS:
    try: pl_df = pl.from_pandas(df)
    except Exception: 
        use_polars, pl_df = False, None
else:
    pl_df = None


# ================= Filtreler ================= #
left, right = st.columns([3, 2])
with left:
    unique_tests = sorted([str(x) for x in df["TETKIK_ISMI"].dropna().unique()])
    selected_tests = st.multiselect("Analiz edilecek tetkikler", options=unique_tests, default=unique_tests)
with right:
    sexes = [str(x) for x in df["CINSIYET"].dropna().unique()]
    chosen_sex = st.multiselect("Cinsiyet filtresi", options=sexes, default=sexes)
    files = [str(x) for x in df["SOURCE_FILE"].dropna().unique()]
    chosen_files = st.multiselect("Dosya filtresi", options=files, default=files)

# --- 99 ile başlayan TCKN filtreleme kontrolü ---
st.markdown("### 🧾 TCKN Filtre Seçimi")

tckn_filter = st.selectbox(
    "TCKN filtrele:",
    ["Hepsi", "Sadece gerçek TCKN", "Sadece 99'lu TCKN"],
    index=1,  # Varsayılan: Sadece gerçek TCKN
    help="99 ile başlayanlar genelde geçici kayıtlardır."
)

work = df.copy()
# --- TCKN filtreleme ---
if "TCKIMLIK_NO" in work.columns:
    tckn_str = work["TCKIMLIK_NO"].astype(str)

    if tckn_filter == "Sadece gerçek TCKN":
        work = work[~tckn_str.str.startswith("99", na=False)]

    elif tckn_filter == "Sadece 99'lu TCKN":
        work = work[tckn_str.str.startswith("99", na=False)]

if chosen_sex:
    work = work[work["CINSIYET"].astype(str).isin(chosen_sex)]
if chosen_files:
    work = work[work["SOURCE_FILE"].astype(str).isin(chosen_files)]
if selected_tests:
    work = work[work["TETKIK_ISMI"].astype(str).isin(selected_tests)]

# Güvence: numeric kopya olsun
work = add_numeric_copy(work)


# ================= VARYANT ÖZETİ (etiketleme) ================= #
A2_KEYS = {"A2/","HbA2","HbA2 (%)","Hb A2","Hb A2 (%)"}
F_KEYS  = {"F/","HbF","HbF (%)","Hb F","Hb F (%)"}
NUMVAR_FROM_TEST = {"C/":"HbC", "D/":"HbD", "E/":"HbE", "S/":"HbS"}

def pick_variant_tag(g: pd.DataFrame) -> str | None:
    g = add_numeric_copy(g.copy())
    g["TETKIK_ISMI"] = g["TETKIK_ISMI"].astype(str)
    
    # YENİ - ADIM 1: Önce CLEAN (temizlenmiş) sütununa bak
    # Kullanıcı "Düzenlenebilir tablo"ya "USV" gibi bir değer yazdıysa,
    # bu, diğer tüm kuralları ezer (en yüksek öncelik).
    clean_col = "ANORMAL_HB_CLEAN"
    if clean_col in g.columns:
        # Bu protokole ait CLEAN değerlerini al
        clean_values = g[clean_col].dropna().astype(str)
        clean_values = clean_values[clean_values != ""]
        if not clean_values.empty:
            # Temizlenmiş değeri (örn. "USV", "HbS", "HbC") doğrudan etiket olarak döndür
            return clean_values.iloc[0] 

    # ADIM 2: CLEAN yoksa, eski mantığa (otomatik sınıflandırma) devam et
    tags = []
    
    # 1) Anormal Hb/ metinlerinden (norm_anormal_hb_text "USV"yi zaten tanıyor olmalı)
    txt = g.loc[g["TETKIK_ISMI"] == "Anormal Hb/", "TEST_DEGERI"].dropna().astype(str)
    for v in txt:
        t = norm_anormal_hb_text(v)
        if t: tags.append(t)
        
    # 2) A2/F erişkin eşikleri
    if g["TETKIK_ISMI"].isin(A2_KEYS).any():
        a2 = g.loc[g["TETKIK_ISMI"].isin(A2_KEYS), "__VAL_NUM__"].dropna()
        if not a2.empty and a2.max() >= 3.5: tags.append("HbA2↑")
    if g["TETKIK_ISMI"].isin(F_KEYS).any():
        f = g.loc[g["TETKIK_ISMI"].isin(F_KEYS), "__VAL_NUM__"].dropna()
        if not f.empty and f.max() > 2.0: tags.append("HbF↑")
        
    # 3) HPLC pikleri
    for k, var_name in NUMVAR_FROM_TEST.items():
        m = g["TETKIK_ISMI"] == k
        if m.any():
            vv = g.loc[m, "__VAL_NUM__"].dropna()
            if not vv.empty and (vv > 0).any():
                tags.append(var_name)
    
    if not tags: return None
    # Öncelik listesi (USV'yi içermeli)
    for p in ["Hb S-β-thal","HbS","HbC","HbD","HbE","USV","HbA2↑","HbF↑","Normal"]:
        if p in tags: return p
    return tags[0]

if "VARIANT_TAG" not in work.columns:
    var_map = (work.groupby("PROTOKOL_NO", group_keys=False)
                  .apply(lambda g: pd.Series({"VARIANT_TAG": pick_variant_tag(g)}))
                  .reset_index())
    work = work.merge(var_map, on="PROTOKOL_NO", how="left")

st.header("📋 Varyant Özeti — erişkin eşikleri ile")
present = [t for t in ["Hb S-β-thal","HbS","HbC","HbD","HbE","HbA2↑","HbF↑","Normal"]
        _if t in set(work["VARIANT_TAG"].dropna())]
variant_choice = st.selectbox("Varyant seç:", ["(Tümü)"] + present, index=0)

base_v = work.copy()
if variant_choice != "(Tümü)":
    base_v = base_v[base_v["VARIANT_TAG"] == variant_choice]

# 1) Tümü için frekans
if variant_choice == "(Tümü)":
    freq = (work["VARIANT_TAG"].value_counts(dropna=True)
            .rename_axis("Varyant").to_frame("N").reset_index())
    total = int(freq["N"].sum()) if not freq.empty else 0
    if total > 0: freq["%"] = (freq["N"]/total*100).round(2)
    st.subheader("Varyant Frekansları")
    st.dataframe(freq, use_container_width=True)
    st.download_button("⬇️ Varyant frekansları (CSV)",
                       data=freq.to_csv(index=False).encode("utf-8-sig"),
                       file_name="varyant_frekans.csv", mime="text/csv")

# 2) Seçilen varyant için ♀/♂ Mean ± SD tablosu
def _mean_sd(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return "—" if s.empty else f"{s.mean():.2f} ± {s.std(ddof=1):.2f}"

# --- YENİ VE GENİŞLETİLMİŞ HALİ ---
PARAMS = {
    # --- Hemogram Parametreleri ---
    "Hemogram/HGB":  ("Hb (g/dL)",    "F: 11–15; M: 12–17"),
    "Hemogram/HCT":  ("HCT (%)",      "F: 36–46; M: 40–53"),
    "Hemogram/RBC":  ("RBC (×10⁶)",   "F: 3.9–5.6; M: 4.5–6.0"),
    "Hemogram/RDW":  ("RDW (%)",      "11–16"),
    "Hemogram/MCV":  ("MCV (fL)",     "80–100"),
    "Hemogram/MCH":  ("MCH (pg)",     "27–34"),
    "Hemogram/MCHC": ("MCHC (g/dL)", "32–36"),
    # --- Buraya diğer hemogram parametrelerini ekleyin (ÖRN) ---
    "Hemogram/PLT":  ("PLT (×10³)",   "150-450"),
    "Hemogram/WBC":  ("WBC (×10³)",   "4.0-11.0"),
    
    # --- HPLC Parametreleri (Mevcut) ---
    "Talasemi(HPLC) (A0)/":           ("HbA0 (%)",      "94–98"),
    "HbA0 (%)":      ("HbA₂ (%)",     "94–98"),
    "A0/":           ("HbA₂ (%)",     "94–98"), # A0 için alternatif isim
    "HbA":           ("HbA (%)",      "94–98"),
    "HbA2 (%)":      ("HbA₂ (%)",     "2–3.5"),
    "A2/":           ("HbA₂ (%)",     "2–3.5"), # A2 için alternatif isim
    "HbF (%)":       ("Hb F (%)",     "0–2"),
    "F/":            ("Hb F (%)",     "0–2"),   # F için alternatif isim
    
    # --- YENİ EKLENEN HPLC VARYANTLARI ---
    "HbS (%)":       ("HbS (%)",      "0"),
    "S/":            ("HbS (%)",      "0"),   # S için alternatif isim
    "HbC (%)":       ("HbC (%)",      "0"),
    "C/":            ("HbC (%)",      "0"),   # C için alternatif isim
    "HbD (%)":       ("HbD (%)",      "0"),
    "D/":            ("HbD (%)",      "0"),   # D için alternatif isim
    "HbE (%)":       ("HbE (%)",      "0"),
    "E/":      _      ("HbE (%)",      "0"),   # E için alternatif isim
    # YENİ EKLENEN USV SATIRI (Eğer verinizde "USV/" gibi bir test ismi varsa)
    "USV/":          ("USV (%)",      "—"),
    "USV (%)":       ("USV (%)",      "—"),
}

table_fm = pd.DataFrame()
if variant_choice != "(Tümü)":
    rows = []
    for tetkik_key, (disp, ref) in PARAMS.items():
        subp = base_v[base_v["TETKIK_ISMI"] == tetkik_key].copy()
        if subp.empty: 
            continue
        subp = add_numeric_copy(subp)  # __VAL_NUM__ güvence
        fem = _mean_sd(subp.loc[subp["CINSIYET"].astype(str).str.lower().str.startswith(("k","f")), "__VAL_NUM__"])
        male= _mean_sd(subp.loc[subp["CINSIYET"].astype(str).str.lower().str.startswith(("e","m")), "__VAL_NUM__"])
        rows.append({"Parameter": disp, "Female (Mean ± SD)": fem, "Male (Mean ± SD)": male, "Reference range": ref})
    table_fm = pd.DataFrame(rows)
    st.subheader("♀/♂ Mean ± SD (seçilen varyant)")
    if table_fm.empty:
        st.info("Bu varyant için parametrik veri bulunamadı.")
    else:
        st.dataframe(table_fm, use_container_width=True)
        st.download_button("⬇️ Tablo #1 (CSV)",
                           data=table_fm.to_csv(index=False).encode("utf-8-sig"),
                           file_name="varyant_ozet_female_male.csv", mime="text/csv")

# 3) Birleşik tablo (Varyant Frekansları + Mean±SD)
if variant_choice != "(Tümü)":
    freq_part = locals().get("freq", pd.DataFrame(columns=["Varyant","N","%"])).copy()
    if not freq_part.empty:
        freq_part = freq_part.rename(columns={"Varyant":"Başlık"})
        freq_part.insert(0,"Bölüm","Varyant Frekansları")
    msd_part = table_fm.copy()
    if not msd_part.empty:
        msd_part = msd_part.rename(columns={"Parameter":"Başlık"})
        msd_part.insert(0,"Bölüm","♀/♂ Mean ± SD")
    cols = ["Bölüm","Başlık","N","%","Female (Mean ± SD)","Male (Mean ± SD)","Reference range"]
    for dfc in (freq_part, msd_part):
        for c in cols:
            if c not in dfc.columns: dfc[c] = None
    combined_df = pd.concat([freq_part[cols], msd_part[cols]], ignore_index=True)
    st.subheader("🧩 Birleşik Tablo")
    st.dataframe(combined_df, use_container_width=True)
    st.download_button("⬇️ Birleşik tablo (CSV)",
                       data=combined_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"birlesik_{variant_choice}.csv",
                       mime="text/csv")


# ================= Kategorik Veri Analizi — Benzersiz Değerler ================= #
st.header("🧬 Kategorik Veri Analizi — Benzersiz Değerler")
for test_name in ["Kan Grubu/", "Anormal Hb/"]:
    sub = work[work["TETKIK_ISMI"].astype(str) == test_name].copy()
    if sub.empty:
        st.warning(f"{test_name} verisi bulunamadı.")
        continue

    st.subheader(f"🔍 {test_name}")

    raw_text = sub["TEST_DEGERI"].astype(str).str.strip()
    if test_name == "Kan Grubu/":
        normalized = raw_text.map(normalize_blood_group)
    else:
        normalized = raw_text.map(norm_anormal_hb_text)

    # ============ ÖZEL AKIŞ: ANORMAL Hb/ ============
    if test_name == "Anormal Hb/":
        # 1) Ham yazım → TC listesi (Frekans yerine)
        sub_nonempty = sub[raw_text.ne("") & sub["TEST_DEGERI"].notna()].copy()
        if sub_nonempty.empty:
            st.info("Anormal Hb/ için dolu metin bulunamadı.")
        else:
            # Her ham değer için benzersiz TCKIMLIK_NO listesini çıkar
            map_tc = (
                sub_nonempty
                .assign(_val=raw_text.loc[sub_nonempty.index])
                .groupby("_val", dropna=False)["TCKIMLIK_NO"]
                .apply(lambda s: ", ".join(sorted({str(x) for x in s.dropna().astype(str)})) or "—")
                .reset_index()
                .rename(columns={"_val": "Ham Değer", "TCKIMLIK_NO": "TCKIMLIK_NO (liste)"})
            )
            st.markdown("**Ham yazımlar (TC listeli)**")
            st.dataframe(map_tc, use_container_width=True)
            st.download_button(
                "⬇️ AnormalHb_ham_yazim_TC_listesi.csv",
                data=map_tc.to_csv(index=False).encode("utf-8-sig"),
                file_name="AnormalHb_ham_yazim_TC_listesi.csv",
                mime="text/csv",
            )

        # 2) Düzenlenebilir tablo (CLEAN kolonu)
        edit_cols = [c for c in ["PROTOKOL_NO","TCKIMLIK_NO","CINSIYET","SOURCE_FILE","TEST_DEGERI"] if c in sub_nonempty.columns]
        edit_df = sub_nonempty[edit_cols].copy()
        clean_col = "ANORMAL_HB_CLEAN"
        # Daha önce varsa koru; yoksa normalize öneriyi doldur
        if clean_col in sub_nonempty.columns:
            edit_df[clean_col] = sub_nonempty[clean_col].astype(str)
        else:
            edit_df[clean_col] = normalized.loc[sub_nonempty.index].fillna("").astype(str)

        st.markdown("**Düzenlenebilir tablo (CLEAN değerini yazın)**")
        edited = st.data_editor(
            edit_df,
            use_container_width=True,
            key="anormalhb_editor",
            column_config={
                "TEST_DEGERI": st.column_config.TextColumn(label="ORIGINAL", help="Ham değer", disabled=True),
                clean_col: st.column_config.TextColumn(label="CLEAN (düzenlenebilir)"),
            },
        )
        col_apply, col_over = st.columns([1,1])
        with col_apply:
            apply_now = st.button("✅ Uygula ve kaydet (oturum içi)", key="apply_anormalhb")
        with col_over:
            overwrite_main = st.checkbox("ORIGINAL sütununu da CLEAN ile değiştir", value=False, key="over_anormalhb")

        # --- DÜZELTİLMİŞ GİRİNTİ BLOKU BAŞLANGICI ---
        if apply_now and not edited.empty:
            # 1. Düzenlenen satırları (edited) al
            upd = edited[[c for c in ["PROTOKOL_NO","TEST_DEGERI",clean_col] if c in edited.columns]].copy()
            upd.rename(columns={clean_col: "__CLEAN_TMP__"}, inplace=True)

            key_proto = work["PROTOKOL_NO"].astype(str) if "PROTOKOL_NO" in work.columns else pd.Series("", index=work.index)
            key_test  = work["TEST_DEGERI"].astype(str).str.strip()
            
            # 2. 'work' dataframe'inde ilgili satırların CLEAN sütununu güncelle
            for _, r in upd.iterrows():
                proto = str(r.get("PROTOKOL_NO",""))
                orig  = str(r.get("TEST_DEGERI","")).strip()
                # Anahtar: Protokol NO ve Orijinal TEST_DEGERI
                mask = (key_proto == proto) & (key_test == orig)
                
                work.loc[mask, clean_col] = r["__CLEAN_TMP__"]
                if overwrite_main:
                    # Orijinali de (isteğe bağlı) değiştir
                    work.loc[mask, "TEST_DEGERI"] = r["__CLEAN_TMP__"]
            
            # 3. YENİ ve ÖNEMLİ: VARIANT_TAG'İ YENİDEN HESAPLA
            #    CLEAN sütunu güncellendiği için, tüm 'work' dataframe'i 
            #    için 'VARIANT_TAG' sütununu SİLİP, yeni 
            #    pick_variant_tag (Adım 1'de güncellenen) fonksiyonu ile 
            #    baştan hesaplatıyoruz.
            st.info("CLEAN değerleri uygulandı, tüm VARIANT_TAG'ler yeniden hesaplanıyor...")
            
            if "VARIANT_TAG" in work.columns:
                work = work.drop(columns="VARIANT_TAG") # Eski tag'leri sil
            
            # Tüm 'work' üzerinden tag'leri yeniden hesapla
            var_map = (work.groupby("PROTOKOL_NO", group_keys=False)
                           .apply(lambda g: pd.Series({"VARIANT_TAG": pick_variant_tag(g)}))
                           .reset_index())
            
            # 'work'e yeni VARIANT_TAG sütununu ekle
            work = work.merge(var_map, on="PROTOKOL_NO", how="left")
            
            # USV Hesaplama bloğunun da (varsa) yeniden çalışması için
            # 'work'ü tekrar güncelliyoruz.
            # (Not: Bu kod, bir sonraki adımdaki 'USV (%)' hesaplamasını
            #  tetiklemez, ancak pivot tabloyu doğru 'work' ile besler.)
            
            st.success("VARIANT_TAG'ler başarıyla güncellendi! Pivot tabloyu kontrol edebilirsiniz.")
            
            # 4. Güncellenmiş veriyi indirme
            st.download_button(
                "⬇️ Güncellenmiş veri (CSV)",
                data=work.to_csv(index=False).encode("utf-8-sig"),
                file_name="guncellenmis_veri_VE_tagler.csv",
                mime="text/csv",
            )
        # --- DÜZELTİLMİŞ GİRİNTİ BLOKU SONU ---

        # 3) Seçince hastanın/protokolün tüm tetkikleri
        # --- DÜZELTİLMİŞ GİRİNTİ ---
        st.markdown("**Hızlı inceleme: bir hasta veya protokol seçin**")
        tcs  = sorted({str(x) for x in sub_nonempty.get("TCKIMLIK_NO", pd.Series(dtype=object)).dropna().astype(str)})
        prot = sorted({str(x) for x in sub_nonempty.get("PROTOKOL_NO", pd.Series(dtype=object)).dropna().astype(str)})

        tabs_sel = st.tabs(["Hasta ile seç", "Protokol ile seç"])
        with tabs_sel[0]:
            if tcs:
                sel_tc = st.selectbox("TCKIMLIK_NO", options=tcs, key="sel_tc_anormalhb")
                proto_for_tc = (
                    sub_nonempty.loc[sub_nonempty["TCKIMLIK_NO"].astype(str) == sel_tc, "PROTOKOL_NO"]
                    .astype(str).unique().tolist()
                ) if "PROTOKOL_NO" in sub_nonempty.columns else []
                all_tests = work[
                    (work["TCKIMLIK_NO"].astype(str) == sel_tc) &
                    (work["PROTOKOL_NO"].astype(str).isin(proto_for_tc))
                ].copy()
                show_cols = [c for c in ["PROTOKOL_NO","TETKIK_ISMI","TEST_DEGERI","CINSIYET","SOURCE_FILE"] if c in all_tests.columns]
                st.dataframe(all_tests[show_cols].sort_values(show_cols[:2]) if not all_tests.empty else all_tests, use_container_width=True)
            else:
                st.info("Seçilebilir hasta yok.")
        with tabs_sel[1]:
            if prot:
                sel_p = st.selectbox("PROTOKOL_NO", options=prot, key="sel_proto_anormalhb")
                all_tests = work[work["PROTOKOL_NO"].astype(str) == sel_p].copy()
                show_cols = [c for c in ["PROTOKOL_NO","TETKIK_ISMI","TEST_DEGERI","CINSIYET","SOURCE_FILE","TCKIMLIK_NO"] if c in all_tests.columns]
                st.dataframe(all_tests[show_cols].sort_values("TETKIK_ISMI") if not all_tests.empty else all_tests, use_container_width=True)
            else:
                st.info("Seçilebilir protokol yok.")

        # Bu özel akışta frekans/ki-kare göstermiyoruz.
        continue  # >>> döngünün geri kalanını Kan Grubu/ için çalıştır

    # ============ STANDART AKIŞ: KAN GRUBU/ (mevcut mantığınız) ============
    # 1) Ham yazımların sayımı
    sub_text = raw_text[raw_text.str.contains(r"[A-Za-zİıÖöÜüÇçŞş]", na=False)]
    if sub_text.empty:
        st.info("Harf içeren veri bulunamadı.")
        value_counts = pd.DataFrame(columns=["Benzersiz Değer","Frekans"])
    else:
        value_counts = (
            sub_text.value_counts(dropna=False)
            .rename_axis("Benzersiz Değer")
            .reset_index(name="Frekans")
        )
    st.markdown("**Ham Yazımlar**")
    st.dataframe(value_counts, use_container_width=True)
    st.download_button(
        f"⬇️ {test_name.strip('/')}_benzersiz_degerler.csv",
        data=value_counts.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{test_name.strip('/')}_benzersiz_degerler.csv",
        mime="text/csv"
    )

    # 2) Normalize edilmiş kategorilerin sayımı
    norm_counts = (
        normalized.value_counts(dropna=False)
        .rename_axis("Kategori (normalize)")
        .reset_index(name="N")
    )
    if not norm_counts.empty:
        totalN = int(norm_counts["N"].sum())
        norm_counts["%"] = (norm_counts["N"] / totalN * 100).round(2)
    else:
        norm_counts = pd.DataFrame(columns=["Kategori (normalize)","N","%"])

    st.markdown("**Normalize Edilmiş Kategoriler**")
    st.dataframe(norm_counts, use_container_width=True)
    st.download_button(
        f"⬇️ {test_name.strip('/')}_normalize_frekans.csv",
        data=norm_counts.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{test_name.strip('/')}_normalize_frekans.csv",
        mime="text/csv"
    )

    # 3) Kategorik genel frekans/ki-kare (normalize etiketle)
    cat_name = "__CAT__"
    sub = sub.assign(**{cat_name: normalized})
    freq_all = (sub[cat_name].value_counts(dropna=False)
                .rename_axis("Kategori").to_frame("N").reset_index())
    totalN = int(freq_all["N"].sum()) if not freq_all.empty else 0
    if totalN:
        freq_all["%"] = (freq_all["N"]/totalN*100).round(2)
    else:
        freq_all["%"] = []
    freq_by_sex = (sub.pivot_table(index=cat_name, columns="CINSIYET",
                                   values="PROTOKOL_NO", aggfunc="count", fill_value=0)
                   .astype(int).reset_index().rename(columns={cat_name:"Kategori"}))
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
    sex_summary = summarize_sex_counts(work)
    st.dataframe(sex_summary, use_container_width=True)
with colB:
    st.write("**Dosyaya Göre Satır & Hasta & Tetkik Sayısı**")
    per_file = work.groupby("SOURCE_FILE").agg(
        N=("PROTOKOL_NO", "size"),
        Hasta_Sayisi=("TCKIMLIK_NO", "nunique"),
        Tetkik_Sayisi=("TETKIK_ISMI", "nunique")
    ).reset_index()
    st.dataframe(per_file, use_container_width=True)
    export_df(per_file, "dosya_bazinda_ozet_filtreli.csv")


# ================= Tetkik Bazlı Analiz (Seçim) ================= #
st.header("📊 Tetkik Bazlı Analiz (Seçim)")
results_rows = []
for test_name in selected_tests:
    # === BEGIN PATCH: overall pool for global stats ===
    overall_pool = []
    # === END PATCH ===
    if test_name in CATEGORICAL_TESTS:
        # Kan Grubu/ ve Anormal Hb/ yukarıda özel blokta analiz edildi
        continue

    sub = work[work["TETKIK_ISMI"].astype(str) == test_name].copy()
    if sub.empty: 
        continue

    use_threshold = st.checkbox(
        f"‘{test_name}’ için erişkin eşiğini uygula",
        value=(test_name in THRESHOLDS),
        key=f"th_{test_name}"
    )
    use_gt_zero  = st.checkbox(
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
        st.warning("Filtre sonrası satır bulunamadı."); 
        continue

    stats_overall = descr_stats_fast(sub_work["__VAL_NUM__"])
    normal_flag   = normality_flag(sub_work["__VAL_NUM__"])
    # Normalite testi (etiket + p)
    norm_label, norm_p_disp = normality_test_with_p(sub_work["__VAL_NUM__"])

    # Genel toplama havuzuna ekle
    overall_pool.extend(pd.to_numeric(sub_work["__VAL_NUM__"], errors="coerce").dropna().tolist())



    by_sex  = (sub_work.groupby("CINSIYET", dropna=False)["__VAL_NUM__"]
               .agg(count="count", mean="mean", std="std", min="min", median="median", max="max")).reset_index()
    by_file = (sub_work.groupby("SOURCE_FILE", dropna=False)["__VAL_NUM__"]
               .agg(count="count", mean="mean", std="std", min="min", median="median", max="max")).reset_index()
    _msg_df = sub_work.rename(columns={"__VAL_NUM__": "VAL"})
    msg, _ = nonparametric_test_by_group(_msg_df, "VAL", "CINSIYET")
    # === BEGIN PATCH: collect values for global stats ===
    overall_pool.extend(pd.to_numeric(_msg_df["VAL"], errors="coerce").dropna().tolist())
    # === END PATCH ===


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

if results_rows:
    st.header("🧾 Toplu Özet Tablosu (Seçili Tetkikler)")
    res_df = pd.DataFrame(results_rows)
    # === BEGIN PATCH: append global total row ===
    if len(overall_pool) > 0:
        overall_stats = descr_stats_fast(pd.Series(overall_pool))
        # N'yi tek tek testlerden de toplayabiliriz ama havuz zaten filtre-sonrası gerçek toplamı temsil ediyor
        overall_row = {
            "TETKIK_ISMI": "GENEL TOPLAM",
            "N": overall_stats["count"],
            "Mean": overall_stats["mean"],
            "Median": overall_stats["median"],
            "Std": overall_stats["std"],
            "Min": overall_stats["min"],
            "Q1": overall_stats["q1"],
            "Q3": overall_stats["q3"],
            "Max": overall_stats["max"],
            "Normalite": norm_label,         
            "p (normalite)": norm_p_disp,     
            "Test": "—",
        }
        res_df = pd.concat([res_df, pd.DataFrame([overall_row])], ignore_index=True)
    # === END PATCH ===

    
    st.dataframe(res_df, use_container_width=True)
    export_df(res_df, name="tetkik_ozet.csv")

# ================= PIVOT: VARYANTLARA GÖRE PARAMETRE ÖZETİ (TABLE 2 - AKILLI v2) ================= #
st.header("🔬 Varyantlara Göre Parametre Özeti (Akıllı Format)")
st.caption("Görseldeki Table 2'ye benzer pivot tablo. Sütun başlıkları gruptaki protokol sayısını (n) içerir.")

# 1. 'PARAMS' sözlüğünde tanımlı testleri (HGB, MCV, A2, F vb.) al
params_to_analyze = list(PARAMS.keys())

# 2. Sütun başlıkları (n=?) için VARYANT_TAG'a sahip benzersiz protokol sayılarını HESAPLA
#    Bunu, 'work' dataframe'i hemogram testlerine göre FİLTRELENMEDEN ÖNCE yaparız
try:
    variant_counts = work[
        work["VARIANT_TAG"].notna() & work["PROTOKOL_NO"].notna()
    ].groupby("VARIANT_TAG")["PROTOKOL_NO"].nunique()
    
    # { "HbS": "HbS (n=15)", "HbE": "HbE (n=13)" } gibi bir harita oluştur
    rename_map = {
        tag: f"{tag} (n={count})" for tag, count in variant_counts.items()
    }
except KeyError:
    st.warning("Varyant sayıları (n=?) hesaplanamadı. PROTOKOL_NO sütunu eksik olabilir.")
    rename_map = {} # Harita boş kalır, yeniden adlandırma yapılmaz

# 3. Pivot tablo için veriyi FİLTRELE
#    - VARIANT_TAG'ı olanları (örn. HbS, HbA2↑)
#    - TETKIK_ISMI, PARAMS listemizde olanları (örn. Hemogram/HGB, HbA2 (%))
#    - Sayısal değeri olanları
data_for_pivot = work[
    work["TETKIK_ISMI"].isin(params_to_analyze) &
    work["VARIANT_TAG"].notna() &
    work["__VAL_NUM__"].notna()
].copy()

if data_for_pivot.empty:
    st.info("Pivot tablo için yeterli veri bulunamadı (Varyantı olan ve hemogram/HPLC parametresi içeren).")
else:
    # 4. AKILLI FORMATLAYICI (Süperskript 'a' ve 'b' eklenmiş)
    def _format_smart_summary_superscript(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce").dropna()
        n = len(s)
        
        if n == 0:
            return "—"
        if n == 1:
            return f"{s.iloc[0]:.2f}"
            
        try:
            norm_label, _ = normality_test_with_p(s)
        except Exception:
            norm_label = "bilinmiyor"
        
        # Yetersiz veri veya non-normal ise: Median [Min-Max] + 'b'
        if norm_label != "normal":
            med = s.median()
            min_val = s.min()
            max_val = s.max()
            return f"{med:.2f} [{min_val:.2f}–{max_val:.2f}]ᵇ"
        
        # Normal ise: Mean ± SD + 'a'
        else:
            mean = s.mean()
            std = s.std(ddof=1)
            if pd.isna(std) or std == 0:
                return f"{mean:.2f}"
            return f"{mean:.2f} ± {std:.2f}ᵃ"

    try:
        # 5. Pivot tabloyu oluştur
        pivot_table = pd.pivot_table(
            data_for_pivot,
            values="__VAL_NUM__",
            index="TETKIK_ISMI",
            columns="VARIANT_TAG",
            aggfunc=_format_smart_summary_superscript, # GÜNCELLENMİŞ fonksiyon
            fill_value="—"
        )

        # 6. Satırları (index) yeniden adlandır ve sırala
        display_map = {k: v[0] for k, v in PARAMS.items()}
        ordered_params_in_table = [
            param_key for param_key in PARAMS.keys() 
            if param_key in pivot_table.index
        ]
        
        if ordered_params_in_table:
            pivot_table_reindexed = pivot_table.loc[ordered_params_in_table]
            pivot_table_reindexed.index = pivot_table_reindexed.index.map(display_map)
            pivot_table_reindexed = pivot_table_reindexed.rename_axis("Parametre")

            # 7. YENİ: Sütunları (n=?) içerecek şekilde yeniden adlandır
            if rename_map:
                 # Sadece tabloda var olan sütunları yeniden adlandır
                existing_cols_to_rename = {
                    col: rename_map[col] for col in pivot_table_reindexed.columns 
                    if col in rename_map
                }
                pivot_table_reindexed = pivot_table_reindexed.rename(
                    columns=existing_cols_to_rename
                )

            # 8. Ekranda göster
            st.dataframe(pivot_table_reindexed, use_container_width=True)
            
            # 9. YENİ: Açıklamayı (footnote) ekle
            st.caption("""
                ᵃ: Normal dağılım gösteren veriler (Mean ± SD)  
                ᵇ: Normal dağılım göstermeyen veya yetersiz veriler (Median [Min–Max])
            """)
            
            # 10. İndirme butonu (güncellenmiş tabloyu indirir)
            csv_data = pivot_table_reindexed.to_csv(index=True).encode("utf-8-sig")
            st.download_button(
                "⬇️ Pivot Tabloyu İndir (CSV)",
                data=csv_data,
          _       file_name="varyant_pivot_ozet_akilli_v2.csv",
                mime="text/csv",
            )
        else:
            st.info("Pivot tablo oluşturuldu ancak `PARAMS` listesiyle eşleşen parametre bulunamadı.")

    except Exception as e:
        st.error(f"Pivot tablo oluşturulurken bir hata oluştu: {e}")
st.caption("Not: Kan Grubu ve Anormal Hb analizleri normalize edilerek hesaplanır; ham yazımlar ayrıca CSV olarak indirilebilir.")
