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
import psutil
from scipy import stats
from concurrent.futures import ThreadPoolExecutor


# ============== Ayarlar ============== #
st.set_page_config(page_title="Tetkik Analiz — Optimize", layout="wide")
# ============== SİSTEM İZLEME ============== #
def sistem_durumu():
    # RAM Kullanımı
    mem = psutil.virtual_memory()
    ram_kullanilan = mem.used / (1024 ** 3) # GB cinsinden
    ram_toplam = mem.total / (1024 ** 3)
    ram_yuzde = mem.percent

    # CPU Kullanımı
    cpu_yuzde = psutil.cpu_percent(interval=1) # 1 saniye ölçüm yapar

    return ram_kullanilan, ram_toplam, ram_yuzde, cpu_yuzde

with st.sidebar:
    st.divider()
    st.markdown("### 🖥️ Sistem Durumu")
    if st.checkbox("Canlı İzle", value=False):
        # Anlık değerleri al
        r_used, r_total, r_perc, c_perc = sistem_durumu()
        
        st.metric("CPU Kullanımı", f"%{c_perc}")
        st.metric("RAM Kullanımı", f"{r_used:.1f} / {r_total:.1f} GB", f"%{r_perc}")
        
        if r_perc > 90:
            st.error("⚠️ RAM Dolmak Üzere!")
        elif c_perc > 90:
            st.warning("🔥 İşlemci Tam Yükte!")
    st.divider()
    
REQ_COLS = ["PROTOKOL_NO", "TCKIMLIK_NO", "TETKIK_ISMI", "TEST_DEGERI", "CINSIYET", "YAS"]

# Kategorik (metin) testler
CATEGORICAL_TESTS = {"Kan Grubu/", "Anormal Hb/"}

# --- Erişkin pozitiflik eşikleri (TETKIK_ISMI anahtarları) ---
THRESHOLDS = {
    "HbA2 (%)": (">=", 3.5),
    "A2/":      (">=", 3.5),
    "HbF (%)":  (">",  2.0),
    "F/":       (">",  2.0),
    "HbS (%)":  (">",  0.0),
    "HbC (%)":  (">",  0.0),
    "HbD (%)":  (">",  0.0),
    "HbE (%)":  (">",  0.0),
}
GT_ZERO_DEFAULT = {
    "HbS (%)","HbC (%)","HbD (%)","HbE (%)","HbF (%)","HbA2 (%)","A2/","F/",
    "C/","D/","E/","S/"
}
VARIANT_NUMERIC_TESTS = {
    "HbS (%)","HbC (%)","HbD (%)","HbE (%)","HbF (%)","HbA2 (%)","Anormal Hb/"
}

# --- YENİ VE GENİŞLETİLMİŞ HALİ ---
PARAMS = {
    # --- YENİ EKLENDİ ---
    "YAS":             ("Yaş (yıl)",    "—"),
    # --- Hemogram Parametreleri ---
    "Hemogram/HGB":  ("Hb (g/dL)",    "F: 11–15; M: 12–17"),
    "Hemogram/HCT":  ("HCT (%)",      "F: 36–46; M: 40–53"),
    "Hemogram/RBC":  ("RBC (×10⁶)",   "F: 3.9–5.6; M: 4.5–6.0"),
    "Hemogram/RDW":  ("RDW (%)",      "11–16"),
    "Hemogram/MCV":  ("MCV (fL)",     "80–100"),
    "Hemogram/MCH":  ("MCH (pg)",     "27–34"),
    "Hemogram/MCHC": ("MCHC (g/dL)", "32–36"),
    # --- Buraya diğer hemogram parametrelerini ekleyin (ÖRN) ---
    "Hemogram/PLT":  ("PLT (×10³)",   "150-450"),
    "Hemogram/WBC":  ("WBC (×10³)",   "4.0-11.0"),
    
    # --- HPLC Parametreleri (Mevcut) ---
    "Talasemi(HPLC) (A0)/":         ("HbA0 (%)",     "94–98"),
    "HbA0 (%)":      ("HbA₂ (%)",     "94–98"),
    "A0/":           ("HbA₂ (%)",     "94–98"), # A0 için alternatif isim
    "HbA":           ("HbA (%)",      "94–98"),
    "HbA2 (%)":      ("HbA₂ (%)",     "2–3.5"),
    "A2/":           ("HbA₂ (%)",     "2–3.5"), # A2 için alternatif isim
    "HbF (%)":       ("Hb F (%)",     "0–2"),
    "F/":            ("Hb F (%)",     "0–2"),   # F için alternatif isim
    
    # --- YENİ EKLENEN HPLC VARYANTLARI ---
    "HbS (%)":       ("HbS (%)",      "0"),
    "S/":            ("HbS (%)",      "0"),   # S için alternatif isim
    "HbC (%)":       ("HbC (%)",      "0"),
    "C/":            ("HbC (%)",      "0"),   # C için alternatif isim
    "HbD (%)":       ("HbD (%)",      "0"),
    "D/":            ("HbD (%)",      "0"),   # D için alternatif isim
    "HbE (%)":       ("HbE (%)",      "0"),
    "E/":            ("HbE (%)",      "0"),   # E için alternatif isim
    # YENİ EKLENEN USV SATIRI (Eğer verinizde "USV/" gibi bir test ismi varsa)
    "USV/":          ("USV (%)",      "—"),
    "USV (%)":       ("USV (%)",      "—"),
}

DISPLAY_LIMIT = 400

MALE_TOKENS   = {"e","erkek","m","male","bay"}
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
    summary["% Satır"]  = (summary["Satır Sayısı"] / total_rows * 100).round(2) if total_rows else np.nan
    summary["% Hasta"] = (summary["Hasta (Benzersiz)"] / total_patients * 100).round(2) if total_patients else np.nan
    summary = summary.reset_index()
    summary = summary[["CINSIYET","Hasta (Benzersiz)","% Hasta","Satır Sayısı","% Satır"]]
    return summary.sort_values("Hasta (Benzersiz)", ascending=False).reset_index(drop=True)


def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    if "TEST_DEGERI" in df.columns:
        df["TEST_DEGERI"] = df["TEST_DEGERI"].astype(str)
    for col in ["CINSIYET", "SOURCE_FILE"]: # "TETKIK_ISMI" buradan kaldırıldı
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
    cv  = (std/mean)*100 if mean!=0 else np.nan
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
    if op == ">":  return series >  cut
    if op == "<=": return series <= cut
    if op == "<":  return series <  cut
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
    
    if re.search(r"\bUSV\b|UNIDENTIFIED|TANIMLANAMAYAN", s): return "USV"
    
    # GÜNCELLENMİŞ BLOK
    if re.search(r"S-?BETA ?0|S ?β0", s): return "Hb S-β0 thal"
    if re.search(r"S-?BETA ?\+|S ?β\+", s): return "Hb S-β+ thal"
    if re.search(r"S-?BETA|S ?β", s): return "Hb S-β-thal" # Genel
    
    if re.search(r"\bHBS\b|S TRAIT|S HET|HBS HET|HBS TAS|S-TASIY", s): return "HbS"
    # ... (kalanı aynı) ...
    if re.search(r"\bHBC\b", s): return "HbC"
    if re.search(r"\bHBD\b", s): return "HbD"
    if re.search(r"\bHBE\b", s): return "HbE"
    if re.search(r"\bA2\b|HBA2", s): return "HbA2↑ (B-thal Trait)" # Etiketi standart hale getirelim
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

# --- TCKN FİLTRESİ ---
st.markdown("### 🧾 TCKN Filtre Seçimi (Gelişmiş)")

tckn_filter = st.selectbox(
    "TCKN filtrele:",
    [
        "Hepsi", 
        "Sadece Gerçek TCKN (11 hane, 99'la başlamayan)", 
        "Sadece Yabancı/Geçici (99'lu veya 11 hane olmayan)"
    ],
    index=1,  # Varsayılan: Sadece gerçek TCKN
    help="Gerçek TCKN: 11 haneli ve 99 ile başlamayan. Yabancı/Geçici: 99 ile başlayan VEYA 11 hane olmayan (Dosya No vb.)."
)

# --- YENİ YAŞ FİLTRESİ ---
st.markdown("### 🎂 Yaş Filtre Seçimi")
age_filter = st.selectbox(
    "Yaş filtresi:",
    ["Tümü", "Sadece 18 yaş ve üstü", "Sadece 18 yaş altı"],
    index=0,  # Varsayılan: Tümü
    help="18 yaş altı (<18), 18 yaş ve üstü (>=18). Evlilik taraması için 18 yaş üstü önerilir."
)

work = df.copy()

# --- YAS SÜTUNUNU FİLTRELEME İÇİN SAYISALA DÖNÜŞTÜR (VE 1'LERİ TEMİZLE) ---
if "YAS" in work.columns:
    # 1'leri (placeholder) NaN yap, sonra sayıya çevir
    work["YAS"] = pd.to_numeric(work["YAS"], errors='coerce').replace(1, np.nan)
else:
    # YAS sütunu yoksa, filtrelemenin hata vermemesi için boş bir NaN sütun oluştur
    work["YAS"] = np.nan 

# --- TCKN filtreleme ---
if "TCKIMLIK_NO" in work.columns:
    # Önce str yap, NaN'ları boş string yap, boşlukları temizle
    tckn_str = work["TCKIMLIK_NO"].astype(str).fillna("").str.strip()
    
    # Kural 1: 11 hane mi?
    is_11_digits = tckn_str.str.len() == 11
    
    # Kural 2: 99 ile mi başlıyor?
    starts_with_99 = tckn_str.str.startswith("99")
    
    # Kural 3: "Gerçek TCKN" maskesi
    # 11 haneli OLMALI VE 99 ile BAŞLAMAMALI
    is_gercek_mask = is_11_digits & (~starts_with_99)
    
    # Kural 4: "Yabancı/Geçici" maskesi
    # 99 ile BAŞLAMALI VEYA 11 hane OLMAMALI
    # (Ayrıca boş olmayanları alalım ki NaN'lar bu gruba girmesin)
    is_yabanci_mask = (starts_with_99 | (~is_11_digits)) & (tckn_str != "")

    if tckn_filter == "Sadece Gerçek TCKN (11 hane, 99'la başlamayan)":
        work = work[is_gercek_mask]

    elif tckn_filter == "Sadece Yabancı/Geçici (99'lu veya 11 hane olmayan)":
        work = work[is_yabanci_mask]
    
    # 'Hepsi' seçeneği için hiçbir şey yapma

# --- Yaş filtreleme ---
if age_filter == "Sadece 18 yaş ve üstü":
    # YAS >= 18 olanları al (NaN olmayanları da otomatik alır)
    work = work[work["YAS"] >= 18]
elif age_filter == "Sadece 18 yaş altı":
    # YAS < 18 olanları al (NaN olmayanları da otomatik alır)
    work = work[work["YAS"] < 18]
# 'Tümü' seçiliyse bir şey yapma


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
F_KEYS  = {"F/","HbF","HbF (%)","Hb F","Hb F (%)"}
NUMVAR_FROM_TEST = {"C/":"HbC", "D/":"HbD", "E/":"HbE", "S/":"HbS"}

def pick_variant_tag(g: pd.DataFrame) -> str | None:
    g = add_numeric_copy(g.copy())
    g["TETKIK_ISMI"] = g["TETKIK_ISMI"].astype(str)
    
    # --- KURAL 0: MANUEL DÜZELTME ---
    clean_col = "ANORMAL_HB_CLEAN"
    if clean_col in g.columns:
        clean_values = g[clean_col].dropna().astype(str)
        clean_values = clean_values[clean_values != ""]
        if not clean_values.empty:
            return clean_values.iloc[0] 

    # --- VERİLERİ TOPLA ---
    def get_val(df, keys):
        if isinstance(keys, str): keys = {keys}
        all_keys = set(keys)
        for k in keys:
            if k in PARAMS: 
                display_name = PARAMS[k][0]
                all_keys.update({p_key for p_key, (disp, ref) in PARAMS.items() if disp == display_name})
        s = df.loc[df["TETKIK_ISMI"].isin(all_keys), "__VAL_NUM__"].dropna()
        return s.max() if not s.empty else np.nan

    # Gerekli parametreleri al (RBC ve HGB eklendi)
    mcv = get_val(g, {"Hemogram/MCV"})
    mch = get_val(g, {"Hemogram/MCH"})
    hgb = get_val(g, {"Hemogram/HGB"}) # Anemi kontrolü için
    rbc = get_val(g, {"Hemogram/RBC"}) # Mentzer İndeksi için
    a2 = get_val(g, {"A2/"}) 
    f = get_val(g, {"F/"})   
    s = get_val(g, {"S/"})   
    a = get_val(g, {"HbA"})  
    c = get_val(g, {"C/"})   
    
    # Güvenli Değerler (NaN kontrolü)
    mcv_val = mcv if pd.notna(mcv) else 999.0
    mch_val = mch if pd.notna(mch) else 999.0
    hgb_val = hgb if pd.notna(hgb) else 99.0
    rbc_val = rbc if pd.notna(rbc) else 0.0
    hba2_val = a2 if pd.notna(a2) else 0.0
    hbf_val = f if pd.notna(f) else 0.0
    hbs_val = s if pd.notna(s) else 0.0
    hbc_val = c if pd.notna(c) else 0.0 
    hba_present = (a > 1.0) if pd.notna(a) else False 
    
    # --- YENİ MANTIKLAR ---
    
    # 1. Mikrositoz / Hipokromi
    has_micro_hypo = (mcv_val < 80) or (mch_val < 27)
    
    # 2. Anemi Kontrolü (Cinsiyete göre HGB eşiği)
    is_anemic = False
    sex_series = g["CINSIYET"].dropna().astype(str).str.upper()
    if not sex_series.empty:
        sex = sex_series.iloc[0]
        # Kadın < 12, Erkek < 13 (DSÖ Kriterleri)
        if sex.startswith(('K', 'F')): 
            is_anemic = (hgb_val < 12.0)
        elif sex.startswith(('E', 'M')): 
            is_anemic = (hgb_val < 13.0)
        else:
            is_anemic = (hgb_val < 12.0) # Bilinmiyorsa güvenli sınır

    # 3. Mentzer İndeksi (MCV / RBC)
    # < 13 : Talasemi lehine
    # > 13 : Demir Eksikliği lehine
    mentzer_index = (mcv_val / rbc_val) if rbc_val > 0 else 0
    
    tags = [] 

 # --- Kural 1: Kompleks Varyantlar (S-Beta Talasemi) ---
    # Bu blok A2 YÜKSEKLİĞİNE baktığı için S-Beta'ları yakalar
    if has_micro_hypo and hba2_val > 3.5 and hbs_val > 50:
        if hba_present: tags.append("Hb S-β+ thal")
        else: tags.append("Hb S-β0 thal")
        
    # --- Kural 2: Orak Hücre Anemisi (HbSS) ---
    # HbA2 NORMAL (veya düşük), HbS ÇOK YÜKSEK
    # Limiti 50 yerine 75 yapmak daha güvenlidir, çünkü SS hastaları genelde %85-95 S olur.
    elif hbs_val > 75 and hba2_val <= 3.5: 
        tags.append("Sickle Cell Anemia (HbSS)")
    
    if has_micro_hypo and hba2_val <= 3.5 and (hbf_val >= 5 and hbf_val <= 20):
        tags.append("δβ-thal Trait")
    # --- YENİ KURAL 1d: BETA TALASEMİ INTERMEDIA / MAJOR ŞÜPHESİ ---
    # Kriter: Mikrositoz VAR ve HbF Çok Yüksek (> %10)
    # (Not: S-Beta yukarıda elendiği için buraya sadece Beta türevleri gelir)
    if hbf_val > 10.0:
         if hba2_val > 3.5:
             tags.append("B-thal Intermedia (High A2/High F)")
         else:
             tags.append("B-thal Intermedia (High F only)")
    
    # SENARYO B: Sınırda İntermedia (Orta F + Ciddi Anemi + Mikrositoz)
    # HbF %5-10 arası ama hasta ciddi anemik (Hb < 9) ve mikrositik ise -> İntermedia lehine
    elif (hbf_val >= 5.0 and hbf_val <= 10.0) and (hgb_val < 9.0) and has_micro_hypo:
        tags.append("B-thal Intermedia? (Mod. F + Severe Anemia)")
        
    if (hbs_val > 0) and (hbc_val > 0) and (not hba_present):
        tags.append("Hb S/C or S/O-Arab?") 

    # --- Kural 2: Metin Bazlı ---
    txt = g.loc[g["TETKIK_ISMI"] == "Anormal Hb/", "TEST_DEGERI"].dropna().astype(str)
    for v in txt:
        t = norm_anormal_hb_text(v) 
        if t: tags.append(t)

    # --- Kural 3: Basit Kantitatif ---
    # A) Klasik A2 Taşıyıcı (> 3.5)
    if hba2_val > 3.5: tags.append("HbA2↑ (B-thal Trait)")
    
    # B) Borderline (3.3 - 3.8 arası)
    # A2 3.3-3.8 arası
    criteria_a = (hba2_val >= 3.3 and hba2_val <= 3.8) and has_micro_hypo
    if criteria_a:
        tags.append("Borderline HbA2")
      
    # C) HPFH
    if hbf_val > 2.0:
        if (not has_micro_hypo) and hbf_val > 5.0: tags.append("HPFH?")
        else: tags.append("HbF↑")

    # D) DEMİR EKSİKLİĞİ ve ALFA TALASEMİ AYRIMI (GELİŞMİŞ)
    # Kriter: Mikrositik/Hipokromik VE Normal A2 VE Normal F
    if has_micro_hypo and hba2_val < 3.3 and hbf_val < 5.0:
        
        # Senaryo 1: Anemik ise (HGB Düşük)
        if is_anemic:
            if mentzer_index > 13:
                tags.append("Iron Deficiency Anemia (Probable)")
            else:
                # Hem anemik hem mentzer < 13 ise karışık/şüpheli
                tags.append("Iron Def./Alpha-thal? (Anemic)")
        
        # Senaryo 2: Anemik Değilse (HGB Normal ama MCV düşük)
        # Bu durum Alfa Talasemi Taşıyıcılığı için çok tipiktir
        else:
            tags.append("Alpha-thal Carrier? (Probable)")

    # --- Diğer Varyantlar ---
    for k, var_name in NUMVAR_FROM_TEST.items():
        val = get_val(g, {k}) 
        if pd.notna(val) and val > 0.1:
            if (var_name == "HbS" or var_name == "HbC") and ("Hb S/C or S/O-Arab?" in tags): continue 
            if var_name == "HbS" and val < 50: tags.append("HbS Trait")
            else: tags.append(var_name)
    
    if not tags: return "Normal (Assumed)" 
    
# --- FİNAL ÖNCELİK LİSTESİ ---
    for p in [
        "Hb S-β0 thal", 
        "Hb S-β+ thal", 
        "Hb S/C or S/O-Arab?", 
        
        # YENİ: Intermedia/Major'ları en üste, S-Beta'nın altına ekledik
        "B-thal Intermedia (High A2/High F)",
        "B-thal Intermedia (High F only)",
        "B-thal Intermedia? (Mod. F + Severe Anemia)",
        
        "δβ-thal Trait",
        "Hb S-β-thal",
        "HbS", "HbC", "HbD", "HbE", "USV",
        "HbS Trait",
        
        "Borderline HbA2",
        "HbA2↑ (B-thal Trait)",
        
        "Iron Deficiency Anemia (Probable)",
        "Iron Def./Alpha-thal? (Anemic)",
        "Alpha-thal Carrier? (Probable)",
        
        "HPFH?", "HbF↑",
        "Normal (Assumed)", "Normal"
    ]:
        if p in tags: return p
            
    return tags[0]
    
if "VARIANT_TAG" not in work.columns:
    var_map = (work.groupby("PROTOKOL_NO", group_keys=False)
                   .apply(lambda g: pd.Series({"VARIANT_TAG": pick_variant_tag(g)}))
                   .reset_index())
    work = work.merge(var_map, on="PROTOKOL_NO", how="left")

# ================= VARYANT ÖZETİ (etiketleme ve istatistik) ================= #
if "VARIANT_TAG" not in work.columns:
    var_map = (work.groupby("PROTOKOL_NO", group_keys=False)
                   .apply(lambda g: pd.Series({"VARIANT_TAG": pick_variant_tag(g)}))
                   .reset_index())
    work = work.merge(var_map, on="PROTOKOL_NO", how="left")

st.header("📋 Varyant Özeti — erişkin eşikleri ile")

# Mevcut varyantları bul
present_tags = sorted([t for t in work["VARIANT_TAG"].dropna().unique()])

# --- YENİ: "TOPLAM BETA GRUBU" TANIMI ---
# Bu liste, toplamak istediğiniz 4 ana grubu içerir
BETA_CARRIER_GROUP = [
    "HbA2↑ (B-thal Trait)", # 1. Klasik
    "Borderline HbA2",      # 2. Sınırda
    "δβ-thal Trait",        # 3. Yüksek F'li Taşıyıcı
    "Hb S-β0 thal",         # 4a. S-Beta
    "Hb S-β+ thal"          # 4b. S-Beta
    "HbF↑"                  # 5. HbF↑
]

# Dropdown seçeneklerini oluştur
custom_options = ["(Tümü)", ">> TOPLAM BETA TAŞIYICI GRUBU (Kombine) <<"] + present_tags
variant_choice = st.selectbox("Varyant seç:", custom_options, index=0)

# --- FİLTRELEME MANTIĞI ---
base_v = work.copy()

if variant_choice == "(Tümü)":
    # Hepsini göster, filtreleme yapma
    pass
elif variant_choice == ">> TOPLAM BETA TAŞIYICI GRUBU (Kombine) <<":
    # Sadece o 4 özel grubu filtrele
    base_v = base_v[base_v["VARIANT_TAG"].isin(BETA_CARRIER_GROUP)]
    st.info(f"Bu grup şu varyantların toplamından oluşmaktadır: {', '.join(BETA_CARRIER_GROUP)}")
else:
    # Tek bir varyant seçildiyse sadece onu filtrele
    base_v = base_v[base_v["VARIANT_TAG"] == variant_choice]


# 1) Frekans Tablosu (Seçime Göre)
freq = (base_v["VARIANT_TAG"].value_counts(dropna=True)
        .rename_axis("Varyant").to_frame("N").reset_index())
total = int(freq["N"].sum()) if not freq.empty else 0
if total > 0: freq["%"] = (freq["N"]/total*100).round(2)

if variant_choice == "(Tümü)":
    st.subheader("Tüm Varyantların Frekansı")
    st.dataframe(freq, use_container_width=True)
    st.download_button("⬇️ Varyant frekansları (CSV)",
                      data=freq.to_csv(index=False).encode("utf-8-sig"),
                      file_name="varyant_frekans.csv", mime="text/csv")
else:
    # Kombine veya tekil seçim yapıldığında da frekansları gösterelim
    st.write(f"**Seçilen Gruptaki Dağılım (Toplam n={total}):**")
    st.dataframe(freq, use_container_width=True)


# 2) Seçilen varyant için ♀/♂ İstatistik Tablosu (Seçmeli Format & Sayılar)

# --- Yardımcı Format Fonksiyonları ---
def fmt(val):
    if pd.isna(val): return "—"
    s = f"{val:.2f}"
    if s.endswith(".00"): return s[:-3]
    return s

def _mean_sd(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty: return "—"
    mean = s.mean()
    std = s.std(ddof=1)
    if pd.isna(std) or std == 0: return fmt(mean)
    return f"{fmt(mean)} ± {fmt(std)}"

def _median_min_max(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty: return "—"
    med = s.median()
    min_v = s.min()
    max_v = s.max()
    return f"{fmt(med)} [{fmt(min_v)}–{fmt(max_v)}]"

table_fm = pd.DataFrame()
if variant_choice != "(Tümü)":
    
    # --- YENİ: BAŞLIK İÇİN HASTA SAYISINI HESAPLA ---
    # base_v 'long format' olduğu için (her test bir satır), benzersiz hasta sayısını bulmalıyız
    unique_pats_stats = base_v[['PROTOKOL_NO', 'CINSIYET']].drop_duplicates(subset=['PROTOKOL_NO'])
    unique_pats_stats['Gender_Clean'] = unique_pats_stats['CINSIYET'].astype(str).map(normalize_sex_label).fillna('Bilinmiyor')
    
    n_stat_total = len(unique_pats_stats)
    n_stat_fem = len(unique_pats_stats[unique_pats_stats['Gender_Clean'] == 'Kadın'])
    n_stat_male = len(unique_pats_stats[unique_pats_stats['Gender_Clean'] == 'Erkek'])
    
    # Başlık Metni
    header_text = f"♀/♂ İstatistikler (Total: {n_stat_total}) [F: {n_stat_fem}, M: {n_stat_male}]"

    st.divider()
    
    # --- Format Seçici ve Başlık ---
    col_head, col_opt = st.columns([2, 2])
    with col_head:
        st.subheader(header_text)
    with col_opt:
        stat_mode = st.radio(
            "Tablo Formatı:",
            ["Ortalama ± Standart Sapma (Mean ± SD)", "Ortanca [Min - Max] (Median [Min-Max])"],
            index=0,
            horizontal=True,
            key="variant_summary_stat_mode",
            label_visibility="collapsed"
        )
    
    # Seçime göre başlıkları ve fonksiyonu belirle
    if "Mean" in stat_mode:
        col_label_f = "Female (Mean ± SD)"
        col_label_m = "Male (Mean ± SD)"
        func_stat = _mean_sd
    else:
        col_label_f = "Female (Median [Min-Max])"
        col_label_m = "Male (Median [Min-Max])"
        func_stat = _median_min_max

    rows = []
    
    # ADIM 1 - YAŞ'ı özel olarak işle
    if "YAS" in base_v.columns:
        age_data = base_v[['PROTOKOL_NO', 'CINSIYET', 'YAS']].dropna(subset=['PROTOKOL_NO', 'YAS']).drop_duplicates(subset=['PROTOKOL_NO'])
        # 1 yaş temizliği
        age_data['YAS'] = pd.to_numeric(age_data['YAS'], errors='coerce').replace(1, np.nan)
        age_data['Gender_Clean'] = age_data['CINSIYET'].astype(str).map(normalize_sex_label).fillna('Bilinmiyor')
        
        fem_age = age_data.loc[age_data['Gender_Clean'] == 'Kadın', "YAS"]
        male_age = age_data.loc[age_data['Gender_Clean'] == 'Erkek', "YAS"]
        
        rows.append({
            "Parameter": "Yaş (yıl)", 
            col_label_f: func_stat(fem_age), 
            col_label_m: func_stat(male_age), 
            "Reference range": PARAMS.get("YAS", ("Yaş", "—"))[1]
        })

    # ADIM 2 - Kalan PARAMS'ları (Hemogram, HPLC) işle
    for tetkik_key, (disp, ref) in PARAMS.items():
        if tetkik_key == "YAS": continue 
            
        subp = base_v[base_v["TETKIK_ISMI"] == tetkik_key].copy()
        if subp.empty: continue
            
        subp = add_numeric_copy(subp)
        subp['Gender_Clean'] = subp['CINSIYET'].astype(str).map(normalize_sex_label).fillna('Bilinmiyor')
        
        fem = subp.loc[subp['Gender_Clean'] == 'Kadın', "__VAL_NUM__"]
        male = subp.loc[subp['Gender_Clean'] == 'Erkek', "__VAL_NUM__"]
        
        rows.append({
            "Parameter": disp, 
            col_label_f: func_stat(fem), 
            col_label_m: func_stat(male), 
            "Reference range": ref
        })
    
    table_fm = pd.DataFrame(rows)
    
    if table_fm.empty:
        st.info("Bu varyant için parametrik veri bulunamadı.")
    else:
        st.dataframe(table_fm, use_container_width=True)
        
        file_suffix = "mean_sd" if "Mean" in stat_mode else "median_minmax"
        st.download_button(
            f"⬇️ Tabloyu İndir (CSV - {file_suffix})",
            data=table_fm.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"varyant_ozet_{variant_choice}_{file_suffix}.csv", 
            mime="text/csv"
        )

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

    # ============ ÖZEL AKIŞ: ANORMAL Hb/ (GÜNCELLENMİŞ v2) ============
    if test_name == "Anormal Hb/":
        
        # 1. YENİ FİLTRE: Kullanıcının istediği gibi, hem 'Anormal Hb/' hem de 'USV/' olanları getir
        #    (ve 'Anormal Hb/'den dönüştürülmüş olabilecek diğerlerini)
        #    Bunu, 'work' dataframe'indeki 'Anormal Hb/'den türeyen tüm varyantları bularak yapalım
        
        # 'norm_anormal_hb_text' fonksiyonunun döndürebileceği tüm olası metin etiketleri
        # (Bu, 'pick_variant_tag' içindeki öncelik listesinden alınabilir)
        known_hb_variants = {"Hb S-β-thal","HbS","HbC","HbD","HbE","USV","HbA2↑","HbF↑","Normal"}
        
        # 'TETKIK_ISMI'si 'Anormal Hb/' OLAN veya 'Anormal Hb/'den DÖNÜŞTÜRÜLMÜŞ olabilecek
        # (örn. 'USV/') satırları göster.
        # En güvenli yol, 'Anormal Hb/' testinin metin içerdiği bilinen satırları almaktır.
        
        # Orijinal 'sub' filtresini koruyalım ve 'sub_nonempty'yi genişletelim
        # sub = work[work["TETKIK_ISMI"].astype(str) == test_name].copy()
        
        # YENİ FİLTRE: Sadece "Anormal Hb/" değil, "USV/" gibi elle düzeltilmiş olanları da göster
        target_tests = {"Anormal Hb/", "USV/"} 
        
        # Eğer 'work' içinde 'ANORMAL_HB_CLEAN' varsa, oradaki değerleri de hedef listeye ekle
        if "ANORMAL_HB_CLEAN" in work.columns:
             target_tests.update(work["ANORMAL_HB_CLEAN"].dropna().unique())
             
        # 'pick_variant_tag' içinde 'USV/' gibi etiketlenen testleri de dahil et
        # Bu çok karmaşık olacağı için şimdilik 'Anormal Hb/' ve 'USV/'ye odaklanalım:
        
        filter_list = {"Anormal Hb/", "USV/"}
        sub = work[work["TETKIK_ISMI"].astype(str).isin(filter_list)].copy()

        # 1) Ham yazım → TC listesi (Frekans yerine) - BU KISIM AYNI KALABİLİR
        sub_nonempty = sub[sub["TEST_DEGERI"].notna() & (sub["TEST_DEGERI"].astype(str).str.strip() != "")].copy()
        
        if sub_nonempty.empty:
            st.info("Düzenlenecek 'Anormal Hb/' veya 'USV/' satırı bulunamadı.")
            # Hızlı inceleme ve diğerleri için bu bloğu atla
        else:
            # 2) Düzenlenebilir tablo (DOĞRUDAN DÜZENLEME)
            # 'CLEAN' sütununu kaldırıyoruz
            edit_cols = [c for c in ["PROTOKOL_NO","TCKIMLIK_NO","CINSIYET","SOURCE_FILE","TETKIK_ISMI","TEST_DEGERI"] if c in sub_nonempty.columns]
            edit_df = sub_nonempty[edit_cols].copy()
            
            # YENİ: Değişiklikleri takip etmek için ana 'work' index'ini bir sütun olarak ekle
            edit_df["__ORIG_INDEX__"] = edit_df.index
            
            st.markdown("**Düzenlenebilir tablo (TETKIK_ISMI ve TEST_DEGERI)**")
            st.caption("Burada 'AnormalHb/' ismini 'USV/' olarak veya 'TEST_DEGERI'ni (örn. 'HBS D LOS ANGELES') 'USV' olarak değiştirebilirsiniz.")
            
            edited = st.data_editor(
                edit_df,
                use_container_width=True,
                key="anormalhb_editor_v2",
                column_config={
                    # YENİ: Bu iki sütun artık düzenlenebilir
                    "TETKIK_ISMI": st.column_config.TextColumn(label="TETKIK_ISMI (düzenlenebilir)"),
                    "TEST_DEGERI": st.column_config.TextColumn(label="TEST_DEGERI (düzenlenebilir)"),
                    
                    # Bu sütunları kilitle
                    "PROTOKOL_NO": st.column_config.TextColumn(disabled=True),
                    "TCKIMLIK_NO": st.column_config.TextColumn(disabled=True),
                    "CINSIYET": st.column_config.TextColumn(disabled=True),
                    "SOURCE_FILE": st.column_config.TextColumn(disabled=True),
                    
                    # YENİ: Index sütununu gizle
                    "__ORIG_INDEX__": None, 
                }
            )
            
            apply_now = st.button("✅ Uygula ve kaydet (oturum içi)", key="apply_anormalhb_v2")

            if apply_now and not edited.empty:
                st.info("Değişiklikler uygulanıyor...")
                
                # 1. Değişiklikleri bulmak için 'edited' ve 'edit_df'yi karşılaştır
                # (Daha basit yöntem: 'edited'deki her satırı 'orig_index' kullanarak 'df' ve 'work'e geri yaz)
                
                update_count = 0
                for _, changed_row in edited.iterrows():
                    orig_index = changed_row["__ORIG_INDEX__"]
                    
                    # Orijinal satırın 'df' ve 'work'te hala var olduğunu kontrol et
                    if orig_index not in df.index or orig_index not in work.index:
                        continue
                        
                    # Yeni değerleri al
                    new_tetkik_ismi = changed_row["TETKIK_ISMI"]
                    new_test_degeri = changed_row["TEST_DEGERI"]
                    
                    # Orijinal değerlerle karşılaştır (gereksiz yazmayı önle)
                    orig_tetkik = work.loc[orig_index, "TETKIK_ISMI"]
                    orig_test_val = work.loc[orig_index, "TEST_DEGERI"]
                    
                    if (orig_tetkik != new_tetkik_ismi) or (orig_test_val != new_test_degeri):
                        update_count += 1
                        
                        # 2. Değişiklikleri ANA 'df'ye uygula (Kalıcılık için)
                        df.loc[orig_index, "TETKIK_ISMI"] = new_tetkik_ismi
                        df.loc[orig_index, "TEST_DEGERI"] = new_test_degeri
                        
                        # 3. Değişiklikleri GEÇİCİ 'work'e uygula (Bu anki görünüm için)
                        work.loc[orig_index, "TETKIK_ISMI"] = new_tetkik_ismi
                        work.loc[orig_index, "TEST_DEGERI"] = new_test_degeri
                        
                        # 4. YENİ: Değişen satırın sayısal değerini de güncelle
                        #    (coerce_numeric fonksiyonu yukarıda tanımlı olmalı)
                        new_val_num = coerce_numeric(pd.Series([new_test_degeri])).iloc[0]
                        work.loc[orig_index, "__VAL_NUM__"] = new_val_num
                        df.loc[orig_index, "__VAL_NUM__"] = new_val_num # Ana df'i de güncelle

                st.info(f"{update_count} satır güncellendi.")

                # 5. YENİ ve ÖNEMLİ: VARIANT_TAG'İ YENİDEN HESAPLA
                st.info("Tüm VARIANT_TAG'ler yeniden hesaplanıyor...")
                if "VARIANT_TAG" in work.columns:
                    work = work.drop(columns="VARIANT_TAG") # Eski tag'leri sil
                if "VARIANT_TAG" in df.columns:
                    df = df.drop(columns="VARIANT_TAG") # Ana df'ten de sil
                
                # 'work' üzerinden tag'leri yeniden hesapla
                var_map_work = (work.groupby("PROTOKOL_NO", group_keys=False)
                               .apply(lambda g: pd.Series({"VARIANT_TAG": pick_variant_tag(g)}))
                               .reset_index())
                work = work.merge(var_map_work, on="PROTOKOL_NO", how="left")
                
                # 'df' üzerinden tag'leri yeniden hesapla
                var_map_df = (df.groupby("PROTOKOL_NO", group_keys=False)
                               .apply(lambda g: pd.Series({"VARIANT_TAG": pick_variant_tag(g)}))
                               .reset_index())
                df = df.merge(var_map_df, on="PROTOKOL_NO", how="left")
                
                st.success("VARIANT_TAG'ler başarıyla güncellendi! Pivot tabloyu kontrol edebilirsiniz.")
                
                # 6. Güncellenmiş veriyi indirme
                st.download_button(
                    "⬇️ Güncellenmiş veri (CSV)",
                    data=work.to_csv(index=False).encode("utf-8-sig"),
                    file_name="guncellenmis_veri_v2.csv",
                    mime="text/csv",
                    key="download_v2"
                )

        # 3) Seçince hastanın/protokolün tüm tetkikleri (Bu kısım aynı kalır)
        st.markdown("**Hızlı inceleme: bir hasta veya protokol seçin**")
        tcs  = sorted({str(x) for x in sub_nonempty.get("TCKIMLIK_NO", pd.Series(dtype=object)).dropna().astype(str)})
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
                
        # --- YENİ: Anormal Hb/ Olan Hastaların TÜM Verilerini İndir ---
        st.divider()
        st.markdown("### 📥 Anormal Hb/ Kaydı Olanların Tüm Verisi")
        st.caption("Aşağıdaki buton, 'Anormal Hb/' testi çalışılmış (sonucu ne olursa olsun) tüm protokollerin, hemogram ve diğer HPLC dahil BÜTÜN sonuçlarını indirir.")
        
        # 1. Anormal Hb/ testi olan protokolleri bul
        # (sub_nonempty zaten bu filtreyi içeriyor ama garantilemek için ana 'work'ten çekelim)
        anormal_hb_protocols = work.loc[work["TETKIK_ISMI"] == "Anormal Hb/", "PROTOKOL_NO"].unique()
        
        if len(anormal_hb_protocols) > 0:
            # 2. Bu protokollerin TÜM verilerini ana 'work' tablosundan çek
            full_patient_data = work[work["PROTOKOL_NO"].isin(anormal_hb_protocols)].copy()
            
            # 3. Okunaklı bir sıralama yap (Önce Protokol, Sonra Tetkik İsmi)
            full_patient_data = full_patient_data.sort_values(by=["PROTOKOL_NO", "TETKIK_ISMI"])
            
            # 4. İndirme butonu
            csv_full_data = full_patient_data.to_csv(index=False).encode("utf-8-sig")
            
            st.download_button(
                label=f"⬇️ {len(anormal_hb_protocols)} Hastanın Tüm Tetkiklerini İndir (CSV)",
                data=csv_full_data,
                file_name="anormal_hb_hastalarinin_tum_verileri.csv",
                mime="text/csv",
                key="btn_download_full_anormal_hb_data"
            )
        else:
            st.info("Anormal Hb/ kaydı olan protokol bulunamadı.")
            
        # -------------------------------------------------------------
        
        # Bu özel akışta frekans/ki-kare göstermiyoruz.
        
        continue  # >>> döngünün geri kalanını Kan Grubu/ için çalıştır
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
        st.warning("Filtre sonrası satır bulunamadı."); 
        continue

    stats_overall = descr_stats_fast(sub_work["__VAL_NUM__"])
    normal_flag   = normality_flag(sub_work["__VAL_NUM__"])
    # Normalite testi (etiket + p)
    norm_label, norm_p_disp = normality_test_with_p(sub_work["__VAL_NUM__"])

    # Genel toplama havuzuna ekle
    overall_pool.extend(pd.to_numeric(sub_work["__VAL_NUM__"], errors="coerce").dropna().tolist())



    by_sex  = (sub_work.groupby("CINSIYET", dropna=False)["__VAL_NUM__"]
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

# ================= PIVOT: VARYANTLARA GÖRE PARAMETRE ÖZETİ (TABLE 2 - v11 - ÇİFTE SAYIM + DETAYLI TOPLAM) ================= #
st.header("🔬 Varyantlara Göre Parametre Özeti")
st.caption("Görseldeki Table 2'ye benzer pivot tablo. NOT: HbA2 değeri >3.5 olan 'Borderline' hastalar, hem Borderline hem de Taşıyıcı sütununa dahil edilmiştir.")

# 1. 'PARAMS' sözlüğünde tanımlı testleri al
params_to_analyze = list(PARAMS.keys())

# --- VERİ HAZIRLIĞI VE ÇİFTE SAYIM MANTIĞI ---
# 1. Ana veriyi al
data_for_pivot_raw = work[
    work["TETKIK_ISMI"].isin(params_to_analyze) &
    work["VARIANT_TAG"].notna() &
    work["__VAL_NUM__"].notna()
].copy()

# 2. YAS verisini ekle
age_data_to_add = pd.DataFrame()
if "YAS" in work.columns:
    age_data = work[['PROTOKOL_NO', 'VARIANT_TAG', 'YAS']].dropna(subset=['PROTOKOL_NO', 'YAS']).drop_duplicates(subset=['PROTOKOL_NO'])
    age_data['TETKIK_ISMI'] = "YAS"
    age_data = age_data.rename(columns={'YAS': '__VAL_NUM__'})
    age_data_to_add = age_data.dropna(subset=['__VAL_NUM__'])

# 3. Ham veriyi birleştir (Çoğaltılmamış hali)
data_for_pivot_base = pd.concat([age_data_to_add, data_for_pivot_raw])

if data_for_pivot_base.empty:
    st.info("Pivot tablo için yeterli veri bulunamadı.")
else:
    # --- BORDERLINE HASTALARINI BUL VE KOPYALA ---
    borderline_protocols = work[work["VARIANT_TAG"] == "Borderline HbA2"]["PROTOKOL_NO"].unique()
    a2_tests = ["A2/", "HbA2 (%)", "Hb A2", "Hb A2 (%)"]
    borderline_a2_values = work[
        (work["PROTOKOL_NO"].isin(borderline_protocols)) & 
        (work["TETKIK_ISMI"].isin(a2_tests))
    ]
    
    double_count_protocols = []
    if not borderline_a2_values.empty:
        double_count_protocols = borderline_a2_values[
            pd.to_numeric(borderline_a2_values["__VAL_NUM__"], errors='coerce') > 3.5
        ]["PROTOKOL_NO"].unique()
    
    if len(double_count_protocols) > 0:
        rows_to_duplicate = data_for_pivot_base[data_for_pivot_base["PROTOKOL_NO"].isin(double_count_protocols)].copy()
        rows_to_duplicate["VARIANT_TAG"] = "HbA2↑ (B-thal Trait)"
        data_for_pivot = pd.concat([data_for_pivot_base, rows_to_duplicate])
        st.info(f"Bilgi: {len(double_count_protocols)} adet 'Borderline' hasta, A2>3.5 olduğu için 'Beta Talasemi Taşıyıcısı' grubuna da eklendi.")
    else:
        data_for_pivot = data_for_pivot_base

    # --- BAŞLIKLARI HAZIRLA (n=?, %...) ---
    rename_map = {}
    try:
        total_unique_patients = work[work["VARIANT_TAG"].notna()]["PROTOKOL_NO"].nunique()
        if total_unique_patients == 0: total_unique_patients = 1
        
        grouped_counts = data_for_pivot.groupby(['VARIANT_TAG', 'CINSIYET'])['PROTOKOL_NO'].nunique().unstack(fill_value=0)
        
        for tag in data_for_pivot["VARIANT_TAG"].unique():
            if tag in grouped_counts.index:
                sub_grp = data_for_pivot[data_for_pivot["VARIANT_TAG"] == tag][["PROTOKOL_NO", "CINSIYET"]].drop_duplicates()
                sub_grp["Sex"] = sub_grp["CINSIYET"].astype(str).map(normalize_sex_label).fillna("Bilinmiyor")
                
                f_count = len(sub_grp[sub_grp["Sex"] == "Kadın"])
                m_count = len(sub_grp[sub_grp["Sex"] == "Erkek"])
                n_grp = len(sub_grp)
                
                pct = (n_grp / total_unique_patients) * 100
                rename_map[tag] = f"{tag} (n={n_grp}, {pct:.1f}%) (F: {f_count}, M: {m_count})"
    except Exception as e:
        st.warning(f"Başlık oluşturma hatası: {e}")

    # --- FORMATLAYICILAR (GÜNCELLENMİŞ - .00 SİLİCİ) ---
    def fmt(val):
        if pd.isna(val): return "—"
        s = f"{val:.2f}"
        if s.endswith(".00"): return s[:-3]
        return s

    def _format_smart_summary_default(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce").dropna()
        n = len(s)
        if n == 0: return "—"
        if n == 1: return fmt(s.iloc[0])
        try: norm_label, _ = normality_test_with_p(s)
        except: norm_label = "bilinmiyor"
        if norm_label != "normal": return f"{fmt(s.median())} [{fmt(s.min())}–{fmt(s.max())}]ᵇ"
        else: 
            mean = s.mean(); std = s.std(ddof=1)
            if pd.isna(std) or std == 0: return fmt(mean)
            return f"{fmt(mean)} ± {fmt(std)}ᵃ"

    def _format_smart_summary_inverted(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce").dropna()
        n = len(s)
        if n == 0: return "—"
        if n == 1: return fmt(s.iloc[0])
        try: norm_label, _ = normality_test_with_p(s)
        except: norm_label = "bilinmiyor"
        if norm_label != "normal":
            mean = s.mean(); std = s.std(ddof=1)
            if pd.isna(std) or std == 0: return fmt(mean)
            return f"{fmt(mean)} ± {fmt(std)}"
        else: return f"{fmt(s.median())} [{fmt(s.min())}–{fmt(s.max())}]"

    def _process_and_display_pivot(pivot_df, table_title, table_key, file_name_suffix):
        display_map = {k: v[0] for k, v in PARAMS.items()}
        ordered_params_in_table = [k for k in PARAMS.keys() if k in pivot_df.index]
        if not ordered_params_in_table: return
        
        final_pivot_table = pivot_df.loc[ordered_params_in_table]
        final_pivot_table.index = final_pivot_table.index.map(display_map)
        final_pivot_table = final_pivot_table.rename_axis("Parametre")
        
        if rename_map:
            cols_to_rename = {c: rename_map[c] for c in final_pivot_table.columns if c in rename_map}
            final_pivot_table = final_pivot_table.rename(columns=cols_to_rename)
            
        # Sıralama: Toplam en sağa
        cols = list(final_pivot_table.columns)
        total_cols = [c for c in cols if "TOPLAM" in str(c)]
        other_cols = [c for c in cols if "TOPLAM" not in str(c)]
        other_cols.sort()
        final_pivot_table = final_pivot_table[other_cols + total_cols]

        st.subheader(table_title)
        st.dataframe(final_pivot_table, use_container_width=True, key=table_key)
        csv_data = final_pivot_table.to_csv(index=True).encode("utf-8-sig")
        st.download_button(f"⬇️ {table_title} İndir (CSV)", data=csv_data, file_name=f"varyant_pivot_ozet_{file_name_suffix}.csv", mime="text/csv", key=f"download_{table_key}")

    try:
        # --- TOPLAM SÜTUNU HESAPLAMA (DETAYLI CİNSİYET İLE) ---
        
        # 1. Benzersiz hasta listesini 'base' (çoğaltılmamış) veriden çek
        unique_patients_all = data_for_pivot_base[['PROTOKOL_NO', 'CINSIYET']].drop_duplicates(subset=['PROTOKOL_NO'])
        
        # 2. Cinsiyetleri normalize et
        unique_patients_all['Sex_Clean'] = unique_patients_all['CINSIYET'].astype(str).map(normalize_sex_label).fillna("Bilinmiyor")
        
        # 3. Sayıları hesapla
        total_n_all = len(unique_patients_all)
        total_f = len(unique_patients_all[unique_patients_all['Sex_Clean'] == 'Kadın'])
        total_m = len(unique_patients_all[unique_patients_all['Sex_Clean'] == 'Erkek'])
        
        # 4. Başlığı oluştur
        total_col_label = f"TOPLAM (n={total_n_all}) (F: {total_f}, M: {total_m})"
        
        # Tablo 1
        pivot_table_default = pd.pivot_table(data_for_pivot, values="__VAL_NUM__", index="TETKIK_ISMI", columns="VARIANT_TAG", aggfunc=_format_smart_summary_default, fill_value="—")
        total_series_1 = data_for_pivot_base.groupby("TETKIK_ISMI")["__VAL_NUM__"].apply(_format_smart_summary_default)
        pivot_table_default[total_col_label] = total_series_1
        
        _process_and_display_pivot(pivot_table_default, "Tablo 1: Akıllı Format (Normal=SDᵃ, Non-Normal=Medianᵇ)", "akilli_format_varsayilan", "akilli")
        st.caption("""ᵃ: Normal dağılım gösteren veriler (Mean ± SD) \nᵇ: Normal dağılım göstermeyen veya yetersiz veriler (Median [Min–Max])""")
        
        st.divider()
        
        # Tablo 2
        pivot_table_inverted = pd.pivot_table(data_for_pivot, values="__VAL_NUM__", index="TETKIK_ISMI", columns="VARIANT_TAG", aggfunc=_format_smart_summary_inverted, fill_value="—")
        total_series_2 = data_for_pivot_base.groupby("TETKIK_ISMI")["__VAL_NUM__"].apply(_format_smart_summary_inverted)
        pivot_table_inverted[total_col_label] = total_series_2
        
        _process_and_display_pivot(pivot_table_inverted, "Tablo 2: İnvert Edilmiş Format (Normal=Median, Non-Normal=SD)", "invert_edilmis_format", "inverted")
        
    except Exception as e:
        st.error(f"Pivot tablo oluşturulurken bir hata oluştu: {e}")
        
# ================= PIVOT HAM VERİ İNDİRME ================= #
st.subheader("🧬 Ham Veri Listesi (Pivot Tablo Grupları)")
st.caption("Yukarıdaki pivot tabloda gördüğünüz varyant gruplarının (örn. 'HbA2↑ (B-thal Trait)', 'HPFH?') ham hasta listesini (TCKN ve tüm parametreler) indirin.")

# 1. Pivot tablolar için kullandığımız ana 'work' verisini alalım
#    Bu veri 'YAS' sütununu ve tüm filtreleri içerir
#    'data_for_pivot'u kullanamayız çünkü o 'long' formatta
#    ve sadece PARAMS'taki testleri içerir. Bize 'work' lazım.

# 2. 'work' dataframe'i tüm gerekli bilgileri (TCKN, VARIANT_TAG, CINSIYET) içerir
#    'VARIANT_TAG' sütunu olmayan satırları (örn. gruplanmamış) çıkaralım
download_df = work[work["VARIANT_TAG"].notna()].copy()

if download_df.empty:
    st.info("İndirilecek etiketlenmiş ham veri bulunamadı.")
else:
    # 3. İndirme için sütunları sıralayalım (Daha okunaklı olması için)
    cols_to_show = [
        "VARIANT_TAG", 
        "PROTOKOL_NO", 
        "TCKIMLIK_NO", 
        "CINSIYET", 
        "YAS", # 'YAŞ' -> 'YAS' olarak düzeltildi
        "TETKIK_ISMI", 
        "TEST_DEGERI", 
        "SOURCE_FILE"
    ]
    # Sadece 'download_df' içinde var olan sütunları seç
    existing_cols = [c for c in cols_to_show if c in download_df.columns]
    
    # Kalan diğer sütunları da sona ekle (örn. __VAL_NUM__)
    other_cols = [c for c in download_df.columns if c not in existing_cols]
    
    final_download_df = download_df[existing_cols + other_cols]
    
    # 4. Varyant Tag'e ve Protokol No'ya göre sırala
    final_download_df = final_download_df.sort_values(by=["VARIANT_TAG", "PROTOKOL_NO"])
    
    # 5. İndirme butonunu oluştur
    csv_data_ham_veri = final_download_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Tüm Varyant Gruplarının Ham Listesini İndir (CSV)",
        data=csv_data_ham_veri,
        file_name="varyant_gruplari_ham_veri_listesi.csv",
        mime="text/csv",
        key="download_ham_veri_pivot"
    )

# ================= BLOK SONU ================= #
# ================= PREVALANS HESAPLAYICI (Epidemiyolojik Karşılaştırma) ================= #
st.divider()
st.subheader("📈 Genişletilmiş Beta Talasemi Prevalansı")
st.caption("Literatürdeki %2'lik orana kıyasla sizin verinizin durumu. Sadece 'HbA2 Yüksekliği' değil, ilişkili diğer gruplar da hesaba katılır.")

# 1. Toplam 'Gerçek' Protokol Sayısı (Payda)
total_n = work["PROTOKOL_NO"].nunique()

if total_n > 0:
    # 2. Alt Grupların Sayılarını Al
    # A. Klasik Taşıyıcılar (Sizin %1.4'ünüz)
    n_classic = len(work[work["VARIANT_TAG"] == "HbA2↑ (B-thal Trait)"])
    
    # B. Sınırda (Borderline) Olanlar
    # (Not: Kodunuzda "Borderline HbA2" etiketi varsa)
    n_border = len(work[work["VARIANT_TAG"] == "Borderline HbA2"])
    
    # C. Delta-Beta Talasemi (Normal A2, Yüksek F)
    n_db = len(work[work["VARIANT_TAG"] == "δβ-thal Trait"])
    
    # D. HbS-Beta Talasemi (Hem S hem Beta geni var)
    n_sb = len(work[work["VARIANT_TAG"].astype(str).str.contains("Hb S-β", na=False)])
    
    # 3. Toplamları Hesapla
    total_carriers = n_classic + n_border + n_db + n_sb
    prevalence = (total_carriers / total_n) * 100
    
    # 4. Sonuçları Göster
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric(
            label="Klasik Taşıyıcı (HbA2 > 3.5)", 
            value=f"{n_classic} kişi", 
            delta=f"%{(n_classic/total_n)*100:.2f}"
        )
        
    with c2:
        st.metric(
            label="Genişletilmiş Toplam (Tümü)", 
            value=f"{total_carriers} kişi", 
            delta=f"%{prevalence:.2f}",
            help="Klasik + Borderline + δβ-thal + S-β thal toplamı"
        )
        
    with c3:
        st.info(f"""
        **Toplamın İçeriği:**
        - Klasik HbA2↑: {n_classic}
        - Borderline: {n_border}
        - δβ-thal: {n_db}
        - S-β thal: {n_sb}
        """)
        
    if prevalence < 1.8:
        st.warning("Toplam oran hala %2'nin altında. 'Iron Def./Alpha-thal?' grubundaki bazı hastalar, A2 değeri baskılanmış (demir eksikliği yüzünden) Beta Talasemi taşıyıcıları olabilir.")
    else:
        st.success("Genişletilmiş oran literatürdeki (~%2) beklentiyle uyumlu görünüyor.")

else:
    st.write("Hesaplanacak veri yok.")
# ================= EK ANALİZ: Beta Talasemi ve HbF Kesişimi (Venn Analizi) ================= #
st.divider()
st.subheader("📊 Beta Talasemi Taşıyıcılarında HbA2 ve HbF Kesişimi")
st.caption("Bu analiz, HbA2 (>3.5) ve HbF (>2.0) yüksekliğinin birlikteliğini gösterir.")

# 1. Gerekli Veriyi Hazırla (Pivotlama)
# Sadece A2 ve F testlerini içeren satırları al
target_tests_a2 = ["A2/", "HbA2 (%)", "Hb A2", "Hb A2 (%)"]
target_tests_f  = ["F/", "HbF (%)", "Hb F", "Hb F (%)"]
all_targets = target_tests_a2 + target_tests_f

subset = work[work["TETKIK_ISMI"].isin(all_targets) & work["__VAL_NUM__"].notna()].copy()

if not subset.empty:
    # Her protokol için tek satır olacak şekilde pivotla
    # (Sütunlar: Test isimleri, Değerler: Sonuçlar)
    pivot_data = subset.pivot_table(
        index="PROTOKOL_NO", 
        columns="TETKIK_ISMI", 
        values="__VAL_NUM__"
    )
    
    # Sütunları birleştir (Birden fazla A2 ismi varsa tek sütunda topla)
    # A2 sütunu oluştur (Mevcut olanların maksimumunu al)
    cols_a2 = [c for c in pivot_data.columns if c in target_tests_a2]
    pivot_data["FINAL_A2"] = pivot_data[cols_a2].max(axis=1)
    
    # F sütunu oluştur
    cols_f = [c for c in pivot_data.columns if c in target_tests_f]
    pivot_data["FINAL_F"] = pivot_data[cols_f].max(axis=1)
    
    # Sadece her iki verisi de (veya en az biri) olanları al
    analysis_df = pivot_data[["FINAL_A2", "FINAL_F"]].dropna(how='all')
    
    # 2. Gruplandırma Mantığı
    # Eşik Değerler
    CUTOFF_A2 = 3.5
    CUTOFF_F = 2.0
    
    # Mantıksal Kontroller
    has_high_a2 = analysis_df["FINAL_A2"] > CUTOFF_A2
    has_high_f  = analysis_df["FINAL_F"]  > CUTOFF_F
    
    # 3. Sayımları Yap
    # Grup 1: Sadece Yüksek A2 (F normal)
    group_only_a2 = analysis_df[has_high_a2 & (~has_high_f)]
    n_only_a2 = len(group_only_a2)
    
    # Grup 2: Sadece Yüksek F (A2 normal)
    group_only_f = analysis_df[(~has_high_a2) & has_high_f]
    n_only_f = len(group_only_f)
    
    # Grup 3: HER İKİSİ DE Yüksek
    group_both = analysis_df[has_high_a2 & has_high_f]
    n_both = len(group_both)
    
    # Toplam "Anormal" Sayısı (Bu 3 grubun toplamı)
    total_abnormal = n_only_a2 + n_only_f + n_both
    
    if total_abnormal > 0:
        # 4. Sonuç Tablosu
        venn_df = pd.DataFrame({
            "Grup Tanımı": [
                f"Sadece Yüksek HbA2 (>{CUTOFF_A2})",
                f"Sadece Yüksek HbF (>{CUTOFF_F})",
                f"HER İKİSİ DE Yüksek (A2>{CUTOFF_A2} ve F>{CUTOFF_F})",
                "TOPLAM (Anormal Bulgusu Olanlar)"
            ],
            "Kişi Sayısı (n)": [n_only_a2, n_only_f, n_both, total_abnormal],
            "Yüzde (%)": [
                f"%{(n_only_a2/total_abnormal)*100:.1f}",
                f"%{(n_only_f/total_abnormal)*100:.1f}",
                f"%{(n_both/total_abnormal)*100:.1f}",
                "100%"
            ]
        })
        
        st.table(venn_df)
        
        # 5. Yorum Cümlesi (Otomatik Oluşturulur)
        st.info(f"**Yorum:** Beta talasemi taşıyıcılığı veya ilişkili hemoglobinopati şüphesi olan {total_abnormal} kişi arasında; "
                f"{n_only_a2} kişi (%{(n_only_a2/total_abnormal)*100:.1f}) sadece yüksek HbA2'ye, "
                f"{n_only_f} kişi (%{(n_only_f/total_abnormal)*100:.1f}) sadece yüksek HbF'ye sahipken, "
                f"**{n_both} kişi (%{(n_both/total_abnormal)*100:.1f}) hem yüksek HbA2 hem de yüksek HbF değerine sahiptir.**")

        # 6. İndirme Butonu (Kesişim Kümesi İçin)
        if n_both > 0:
            both_high_protocols = group_both.index.tolist()
            both_high_data = work[work["PROTOKOL_NO"].isin(both_high_protocols)].copy()
            
            csv_both = both_high_data.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Hem A2 Hem F Yüksek Olanları İndir (CSV)",
                data=csv_both,
                file_name="hem_a2_hem_f_yuksek_hastalar.csv",
                mime="text/csv"
            )

    else:
        st.info("Veri setinde A2 veya F yüksekliği olan kayıt bulunamadı.")
else:
    st.warning("Analiz için gerekli A2 veya F verisi bulunamadı.")
# ================= DEBUG: HbA2 Grubunda HbF Dedektifi ================= #
st.divider()
st.subheader("🕵️ HbA2↑ Grubunda HbF Dedektifi")

target_group = "HbA2↑ (B-thal Trait)"
# HbF için olası isimleri kontrol et
f_cols = ["HbF (%)", "F/", "Hb F", "Hb F (%)"]

# Bu gruba girenleri bul
indices = work[work["VARIANT_TAG"] == target_group].index
subset = work.loc[indices].copy()

# Long formatta (sizin yapınızda) analiz:
hbf_data = subset[subset["TETKIK_ISMI"].isin(f_cols)].copy()

if not hbf_data.empty:
    hbf_values = pd.to_numeric(hbf_data["__VAL_NUM__"], errors='coerce').dropna()
    
    if not hbf_values.empty:
        # YENİ: %2'den yüksek olanları say
        high_f_count = (hbf_values > 2.0).sum()
        total_count_f = len(hbf_values)
        high_f_ratio = (high_f_count / total_count_f) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**İstatistikler ({target_group}):**")
            st.write(f"Min: {hbf_values.min()}")
            st.write(f"Max: {hbf_values.max()}")
            st.write(f"Medyan: {hbf_values.median()}")
            st.write(f"Ortalama: {hbf_values.mean():.2f}")
            # YENİ GÖSTERİM:
            st.metric(label="HbF > %2 Olan Hasta Sayısı", value=f"{high_f_count} / {total_count_f}", delta=f"%{high_f_ratio:.1f}")
        
        with col2:
            st.write("**En Yüksek 10 HbF Değeri:**")
            st.dataframe(hbf_values.sort_values(ascending=False).head(10))
            
        if hbf_values.max() > 5.0:
            st.warning(f"⚠️ Dikkat: Bu grupta %{hbf_values.max()} gibi yüksek HbF değerleri var. Bu hastalar standart sapmayı yükseltiyor.")
            st.info("Bu hastaların hem A2'si yüksek hem F'si yüksek olduğu için 'HbA2↑' grubuna düştüler.")
    else:
        st.warning("HbF değerleri sayıya çevrilemedi.")
else:
    st.warning("Bu grupta HbF tetkiki bulunamadı.")

# ================= EK ANALİZ: Beta Talasemi Taşıyıcılarında (HbA2↑) Normal MCV & MCH Analizi ================= #
st.divider()
st.subheader("🩸 Beta Talasemi Taşıyıcılarında (HbA2↑) Normal MCV & MCH Analizi")

# 1. Hedef kitleyi belirle (Sadece A2 Taşıyıcıları)
target_tag = "HbA2↑ (B-thal Trait)"
subset_indices = work[work["VARIANT_TAG"] == target_tag].index

if not subset_indices.empty:
    # 2. Sadece bu gruba ait ve sadece MCV/MCH satırlarını al
    relevant_tests = ["Hemogram/MCV", "Hemogram/MCH"]
    
    # Ana veriden (work) ilgili satırları çek
    subset_data = work.loc[subset_indices]
    subset_data = subset_data[subset_data["TETKIK_ISMI"].isin(relevant_tests) & subset_data["__VAL_NUM__"].notna()]

    if not subset_data.empty:
        # 3. Veriyi Pivotla: Her protokol bir satır olsun, MCV ve MCH sütun olsun
        try:
            pivot_check = subset_data.pivot_table(
                index="PROTOKOL_NO", 
                columns="TETKIK_ISMI", 
                values="__VAL_NUM__"
            )
            
            # Kolon isimlerini kontrol et ve sadeleştir
            # (Dosyanızdaki isimler 'Hemogram/MCV' şeklindeyse)
            if "Hemogram/MCV" in pivot_check.columns and "Hemogram/MCH" in pivot_check.columns:
                pivot_check = pivot_check.rename(columns={"Hemogram/MCV": "MCV", "Hemogram/MCH": "MCH"})
                
                # Sadece HER İKİSİ DE (MCV ve MCH) ölçülmüş olanları al
                valid_data = pivot_check.dropna(subset=["MCV", "MCH"])
                
                if not valid_data.empty:
                    # 4. Kuralı Uygula: Normal = MCV >= 80 VE MCH >= 27
                    normal_indices = valid_data[
                        (valid_data["MCV"] >= 80) & 
                        (valid_data["MCH"] >= 27)
                    ].index # Bu indexler PROTOKOL_NO'dur
                    
                    count_normal = len(normal_indices)
                    count_total = len(valid_data)
                    count_micro_hypo = count_total - count_normal
                    
                    # 5. Tabloyu Oluştur
                    summary_df = pd.DataFrame({
                        "Durum": [
                            "Normal İndeksler (MCV≥80 ve MCH≥27)", 
                            "Mikrositik/Hipokromik (MCV<80 veya MCH<27)", 
                            "TOPLAM (Verisi Tam Olanlar)"
                        ],
                        "Hasta Sayısı": [count_normal, count_micro_hypo, count_total],
                        "Oran (%)": [
                            f"{(count_normal/count_total)*100:.1f}%", 
                            f"{(count_micro_hypo/count_total)*100:.1f}%", 
                            "100%"
                        ]
                    })

                    st.write(f"**{target_tag}** grubunda hem MCV hem MCH değeri bulunan hastaların analizi:")
                    st.table(summary_df)
                    
                    # 6. İndirme Butonu (Sadece Normal Olanlar)
                    if count_normal > 0:
                        normal_patients_full = work[work["PROTOKOL_NO"].isin(normal_indices)].copy()
                        
                        csv_normal = normal_patients_full.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "⬇️ Normal İndeksli (Sessiz?) Taşıyıcıları İndir (CSV)",
                            data=csv_normal,
                            file_name="normal_indeksli_b_thal_tasiyicilari.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("HbA2 taşıyıcılarında eşleşen MCV ve MCH verisi bulunamadı.")
            else:
                st.warning("Bu grupta MCV veya MCH testlerinden biri eksik veya isimleri farklı.")
                
        except Exception as e:
            st.error(f"İndeks analizi sırasında hata: {e}")
    else:
        st.warning(f"'{target_tag}' grubu için MCV/MCH verisi bulunamadı.")
else:
    st.info(f"Veri setinde '{target_tag}' grubuna giren hasta bulunamadı.")
# ================= YENİ: INTERMEDIA vs TRAIT AYRIM GRAFİĞİ (ÖZELLEŞTİRİLEBİLİR) ================= #
st.divider()
st.subheader("📉 Klinik Ayrım: Beta Talasemi Minör vs İntermedia")
st.caption("Bu grafik, 'Minör' ve 'İntermedia' şüphesi olan hastaları **Hemoglobin (HGB)** ve **Fetal Hemoglobin (HbF)** düzeylerine göre ayırır.")

# --- GRAFİK AYARLARI (BURAYI DEĞİŞTİREBİLİRSİNİZ) ---
GRAPH_TITLE = "Beta Talasemi Ayrımı: HbF vs Hemoglobin Dağılımı"
X_LABEL = "Fetal Hemoglobin (HbF) %"
Y_LABEL = "Toplam Hemoglobin (HGB) g/dL"

# Renk Paleti (İstediğiniz renkleri buraya yazabilirsiniz)
CUSTOM_COLORS = {
    "HbA2↑ (B-thal Trait)": "blue",        # Klasik Taşıyıcılar (Mavi)
    "Borderline HbA2": "cyan",             # Sınırda Olanlar (Açık Mavi/Turkuaz)
    
    "B-thal Intermedia (High A2/High F)": "red",      # İntermedia Şüphesi (Kırmızı)
    "B-thal Intermedia (High F only)": "darkred",     # İntermedia Şüphesi (Koyu Kırmızı)
    "B-thal Intermedia? (Mod. F + Severe Anemia)": "orange", # Sınırda İntermedia (Turuncu)
    
    "δβ-thal Trait": "green"               # Delta-Beta (Yeşil)
}
# -------------------------------------------------------

# 1. Analiz edilecek grupları belirle (Renk listesinden otomatik alır)
target_variants = list(CUSTOM_COLORS.keys())

# 2. Veriyi Hazırla
hgb_tests = ["Hemogram/HGB"]
f_tests   = ["F/", "HbF (%)", "Hb F", "Hb F (%)"]
relevant_tests = hgb_tests + f_tests

# İlgili verileri ana tablodan çek
subset_graph = work[
    work["TETKIK_ISMI"].isin(relevant_tests) & 
    work["VARIANT_TAG"].isin(target_variants)
].copy()

if not subset_graph.empty:
    # Pivotla (Her protokol tek satır: Sütunlar -> HGB, HbF)
    subset_graph["TYPE"] = subset_graph["TETKIK_ISMI"].apply(lambda x: "HGB" if x in hgb_tests else "HbF")
    
    # Sayısal değere çevir (Garanti olsun)
    subset_graph["__VAL_NUM__"] = pd.to_numeric(subset_graph["__VAL_NUM__"], errors='coerce')
    
    pivot_graph = subset_graph.pivot_table(
        index=["PROTOKOL_NO", "VARIANT_TAG"], 
        columns="TYPE", 
        values="__VAL_NUM__"
    ).reset_index()

    # Hem HGB hem HbF verisi olanları al (Yoksa grafik çizilemez)
    graph_data = pivot_graph.dropna(subset=["HGB", "HbF"])

    if not graph_data.empty:
        # 3. Grafiği Çiz (Matplotlib)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Her grup için ayrı ayrı noktaları (scatter) çiz
        for var_name in graph_data["VARIANT_TAG"].unique():
            sub_g = graph_data[graph_data["VARIANT_TAG"] == var_name]
            
            # Renkleri ayarlarımızdan al
            c = CUSTOM_COLORS.get(var_name, "gray")
            
            # Minörleri biraz daha şeffaf ve küçük yap (Kalabalık oldukları için)
            is_trait = "Trait" in var_name or "Borderline" in var_name
            alpha = 0.5 if is_trait else 0.9
            size = 30 if is_trait else 80 
            
            ax.scatter(sub_g["HbF"], sub_g["HGB"], label=var_name, color=c, alpha=alpha, s=size, edgecolors='w')

        # 4. Referans Çizgileri ve Yazılar
        
        # HGB < 10 çizgisi (Yatay - Anemi Sınırı)
        ax.axhline(y=10, color='black', linestyle='--', linewidth=1)
        
        # HbF > 10 çizgisi (Dikey - İntermedia Sınırı)
        ax.axvline(x=10, color='black', linestyle='--', linewidth=1)
        
        # Bölgeleri İsimlendir
        # Sağ Alt Köşe: Düşük HGB, Yüksek F -> INTERMEDIA
        ax.text(graph_data["HbF"].max(), graph_data["HGB"].min(), "İNTERMEDİA BÖLGESİ", 
                ha='right', va='bottom', fontsize=10, fontweight='bold', color='red', alpha=0.5)
        
        # Sol Üst Köşe: Yüksek HGB, Düşük F -> MINOR
        ax.text(0, graph_data["HGB"].max(), "MİNÖR (TAŞIYICI) BÖLGESİ", 
                ha='left', va='top', fontsize=10, fontweight='bold', color='blue', alpha=0.5)

        # Eksen İsimleri ve Başlık (Sizin Ayarlarınızdan)
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL)
        ax.set_title(GRAPH_TITLE)
        
        # --- İSTEĞE BAĞLI: Eksen Aralıklarını Elle Ayarlamak İçin Yorumu Kaldırın ---
        # ax.set_xlim(0, 20)  # HbF 0 ile 20 arası
        # ax.set_ylim(5, 18)  # HGB 5 ile 18 arası
        # --------------------------------------------------------------------------

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Lejantı dışarı al
        ax.grid(True, alpha=0.3) # Izgara çizgileri

        st.pyplot(fig)
        
        st.info("""
        **Grafik Yorumu:**
        * **Kırmızı/Turuncu Noktalar:** Genellikle sağ alt köşede toplanır (HbF Yüksek, Hb Düşük). Bunlar klinik olarak daha şiddetli (İntermedia) olgulardır.
        * **Mavi/Yeşil Noktalar:** Sol üstte toplanır. Bunlar klasik taşıyıcılardır.
        """)
    else:
        st.warning("Grafik çizmek için seçilen gruplarda hem HGB hem de HbF sonucu olan hasta bulunamadı.")
else:
    st.info("Listede grafiklenecek varyant grubu (A2, Borderline, Intermedia, Delta-Beta) verisi bulunamadı.")
st.caption("Not: Kan Grubu ve Anormal Hb analizleri normalize edilerek hesaplanır; ham yazımlar ayrıca CSV olarak indirilebilir.")

# ================================================================================= #
#             🤖 MAKİNE ÖĞRENMESİ (ML) MODÜLÜ (HEDEF SEÇİMİ + GÜVEN SKORU)          #
# ================================================================================= #
st.divider()
st.header("🤖 Yapay Zeka (ML) Laboratuvarı")
st.caption("Modelin girdilerini (Parametreler) ve çıktılarını (Hastalık Tanıları) seçerek özelleştirilmiş eğitim yapın.")

# --- ML Modülünü Aktif Et ---
if st.checkbox("Yapay Zeka Laboratuvarını Aç", value=False):
    
    # 1. GENİŞLETİLMİŞ PARAMETRE LİSTESİ
    HEMO_PARAMS = [
        "Hemogram/HGB", "Hemogram/RBC", "Hemogram/MCV", "Hemogram/MCH", "Hemogram/MCHC",
        "Hemogram/RDW", "Hemogram/RDW-SD", 
        "Hemogram/HCT", "Hemogram/PLT", "Hemogram/WBC", "Hemogram/MPV", "Hemogram/PCT", "Hemogram/PDW",
        "Hemogram/NEU", "Hemogram/NEU%", "Hemogram/LYM", "Hemogram/LYM%",
        "Hemogram/MONO", "Hemogram/MONO%", "Hemogram/EOS", "Hemogram/EOS%",
        "Hemogram/BASO", "Hemogram/BASO%"
    ]
    
    HPLC_PARAMS = [
        "HbA2 (%)", "A2/", 
        "HbF (%)", "F/", 
        "HbS (%)", "S/", 
        "HbC (%)", "C/", 
        "HbD (%)", "D/", 
        "HbE (%)", "E/",
        "Talasemi(HPLC) (A0)/", "HbA", "HbA (%)"
    ]
    
    OTHER_PARAMS = ["YAS", "CINSIYET"] 
    ALL_AVAILABLE_PARAMS = HEMO_PARAMS + HPLC_PARAMS

    # 2. Kullanıcı Arayüzü
    col_ml_settings, col_ml_main = st.columns([1, 2])
    
    with col_ml_settings:
        st.subheader("⚙️ Model Ayarları")
        
        # A) Algoritma Seçimi
        algo_choice = st.radio(
            "Algoritma Seçin:",
            ["Random Forest", "XGBoost", "LightGBM", "CatBoost"],
            index=1
        )
        
        st.divider()
        
        # B) Parametre Seçimi (GİRDİLER / X)
        st.write("**1. GİRDİLER: Hangi verilerle tahmin yapılsın?**")
        feature_mode = st.radio(
            "Parametre Grubu:",
            ["Tümü (Full Hemogram + HPLC + Yaş/Cinsiyet)", "Sadece Hemogram", "Sadece HPLC", "Özel Seçim"],
            index=0
        )
        
        if feature_mode == "Tümü (Full Hemogram + HPLC + Yaş/Cinsiyet)":
            selected_features = ALL_AVAILABLE_PARAMS
        elif feature_mode == "Sadece Hemogram":
            selected_features = HEMO_PARAMS
        elif feature_mode == "Sadece HPLC":
            selected_features = HPLC_PARAMS
        else:
            selected_features = st.multiselect("Parametreleri İşaretleyin:", ALL_AVAILABLE_PARAMS, default=ALL_AVAILABLE_PARAMS)
            
        st.divider()

        # C) Hedef Seçimi (ÇIKTILAR / y) -- YENİ ÖZELLİK --
        st.write("**2. ÇIKTILAR: Hangi tanıları tahmin edeyim?**")
        
        # Mevcut tüm etiketleri bul
        if "VARIANT_TAG" in work.columns:
            available_targets = sorted(work["VARIANT_TAG"].dropna().unique().tolist())
            
            # Varsayılan olarak hepsini seç, ama kullanıcı çıkarabilir
            selected_targets = st.multiselect(
                "Tahmin Edilecek Hastalık Gruplarını Seçin:",
                options=available_targets,
                default=available_targets,
                help="Listeden çıkardığınız hastalık grubuna sahip hastalar, model eğitimine ve testine DAHİL EDİLMEYECEKTİR."
            )
            
            st.caption(f"Seçili Gruplar: {len(selected_targets)} / {len(available_targets)}")
        else:
            st.error("Önce yukarıdaki analizlerin tamamlanması gerekir.")
            selected_targets = []
        
        st.divider()
        # D) Başlat Butonu
        start_training = st.button("🚀 Modeli Eğit ve Test Et", type="primary", use_container_width=True)

    # 3. Eğitim ve Analiz Süreci
    with col_ml_main:
        if start_training:
            if not selected_features:
                st.error("Lütfen en az bir parametre seçin.")
            elif not selected_targets:
                st.error("Lütfen en az bir hastalık grubu seçin.")
            elif len(selected_targets) < 2:
                st.error("Sınıflandırma yapabilmek için en az 2 farklı grup seçmelisiniz.")
            elif "VARIANT_TAG" not in work.columns:
                st.error("Veri hazırlanmamış.")
            else:
                # --- VERİ HAZIRLIĞI ---
                # Sadece seçili hedef gruplara (selected_targets) ait satırları al
                labeled_data = work[work["VARIANT_TAG"].isin(selected_targets)].copy()
                
                if labeled_data.empty:
                    st.error("Seçilen filtrelere uygun veri kalmadı.")
                else:
                    try:
                        # Kütüphaneler
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
                        from sklearn.preprocessing import LabelEncoder
                        import seaborn as sns
                        
                        # Algoritma Yükleyicileri
                        models = {}
                        try: from sklearn.ensemble import RandomForestClassifier; models["Random Forest"] = RandomForestClassifier
                        except: pass
                        try: from xgboost import XGBClassifier; models["XGBoost"] = XGBClassifier
                        except: pass
                        try: from lightgbm import LGBMClassifier; models["LightGBM"] = LGBMClassifier
                        except: pass
                        try: from catboost import CatBoostClassifier; models["CatBoost"] = CatBoostClassifier
                        except: pass

                        if algo_choice not in models:
                            st.error(f"{algo_choice} kütüphanesi yüklü değil.")
                            st.stop()

                        with st.spinner("Veri hazırlanıyor..."):
                            # 1. Pivotlama (Uzun -> Geniş)
                            ml_subset = labeled_data[
                                labeled_data["TETKIK_ISMI"].isin(selected_features) & 
                                labeled_data["__VAL_NUM__"].notna()
                            ].copy()
                            
                            X = ml_subset.pivot_table(index="PROTOKOL_NO", columns="TETKIK_ISMI", values="__VAL_NUM__")
                            
                            # 2. YAŞ Ekleme
                            if "YAS" in work.columns:
                                age_series = labeled_data.drop_duplicates("PROTOKOL_NO").set_index("PROTOKOL_NO")["YAS"]
                                X = X.join(age_series, how="left")
                                X["YAS"] = X["YAS"].fillna(0)

                            # 3. CİNSİYET Ekleme (0/1 Dönüşümü)
                            if "CINSIYET" in work.columns:
                                sex_series = labeled_data.drop_duplicates("PROTOKOL_NO").set_index("PROTOKOL_NO")["CINSIYET"]
                                sex_series = sex_series.astype(str).map(lambda x: 1 if x.lower().startswith(('e','m')) else 0)
                                X["CINSIYET_CODE"] = sex_series

                            # Eksikleri 0 ile doldur
                            X = X.fillna(0)
                            X = X.astype(float)
                            
                            # Sütun İsimlerini Temizle
                            new_cols = []
                            for col in X.columns:
                                clean_col = str(col).replace("%", "Pct").replace("/", "_").replace(" ", "_").replace("-", "_")
                                clean_col = re.sub(r'[^A-Za-z0-9_]', '', clean_col)
                                new_cols.append(clean_col)
                            X.columns = new_cols

                            # Hedef (y)
                            y_raw = labeled_data.drop_duplicates("PROTOKOL_NO").set_index("PROTOKOL_NO")["VARIANT_TAG"]
                            
                            # Eşleştirme
                            common = X.index.intersection(y_raw.index)
                            X = X.loc[common]
                            y_raw = y_raw.loc[common]
                            
                            # Yetersiz Sınıfları Temizle (<2 örnek)
                            vc = y_raw.value_counts()
                            valid_classes = vc[vc >= 2].index
                            if len(vc[vc < 2]) > 0:
                                st.warning(f"⚠️ Şu nadir tanılar (<2 hasta) eğitimden çıkarıldı: {list(vc[vc < 2].index)}")
                            
                            X = X[y_raw.isin(valid_classes)]
                            y_raw = y_raw[y_raw.isin(valid_classes)]
                            
                            # Label Encoding
                            le = LabelEncoder()
                            y = le.fit_transform(y_raw)
                            class_names = le.classes_
                        
                        # --- MODEL EĞİTİMİ ---
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                        
                        clf = None
                        if algo_choice == "Random Forest":
                            clf = models[algo_choice](n_estimators=100, random_state=42)
                        elif algo_choice == "XGBoost":
                            clf = models[algo_choice](eval_metric='mlogloss', random_state=42)
                        elif algo_choice == "LightGBM":
                            clf = models[algo_choice](random_state=42, verbose=-1)
                        elif algo_choice == "CatBoost":
                            clf = models[algo_choice](verbose=0, random_state=42)

                        with st.spinner(f"{algo_choice} modeli eğitiliyor..."):
                            clf.fit(X_train, y_train)
                            y_pred = clf.predict(X_test)
                            
                            # Olasılıkları al (Güven Skoru için)
                            try:
                                y_proba = clf.predict_proba(X_test)
                                confidence = np.max(y_proba, axis=1) * 100
                            except:
                                confidence = [0] * len(y_pred)

                            acc = accuracy_score(y_test, y_pred)
                        
                        # --- SONUÇ EKRANI ---
                        st.success(f"✅ **{algo_choice}** Başarı Oranı: **%{acc*100:.2f}**")
                        
                        # Sekmeler
                        tab_imp, tab_cm, tab_rep, tab_pred = st.tabs([
                            "📊 Özellik Önemi", 
                            "🎯 Karmaşıklık Matrisi", 
                            "📝 Detaylı Rapor",
                            "🔍 Tahmin Sonuçları (Güven Skoru)"
                        ])
                        
                        with tab_imp:
                            try:
                                importances = clf.feature_importances_
                                feature_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
                                fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
                                feature_imp.head(20).plot.bar(ax=ax_imp, color="#87CEEB")
                                ax_imp.set_title(f"{algo_choice} İçin En Önemli 20 Parametre")
                                plt.xticks(rotation=45, ha='right')
                                st.pyplot(fig_imp)
                            except: st.warning("Çizilemedi.")

                        with tab_cm:
                            unique_indices = sorted(list(set(y_test) | set(y_pred)))
                            unique_names_present = [class_names[i] for i in unique_indices]
                            fig_cm, ax_cm = plt.subplots(figsize=(12, 8))
                            cm = confusion_matrix(y_test, y_pred, labels=unique_indices)
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                        xticklabels=unique_names_present, yticklabels=unique_names_present, ax=ax_cm)
                            plt.ylabel('Gerçek Tanı')
                            plt.xlabel('Modelin Tahmini')
                            plt.xticks(rotation=90)
                            st.pyplot(fig_cm)

                        with tab_rep:
                            report = classification_report(y_test, y_pred, labels=unique_indices, target_names=unique_names_present, output_dict=True, zero_division=0)
                            st.dataframe(pd.DataFrame(report).transpose())
                            
                        with tab_pred:
                            # --- GÜVEN SKORU ANALİZİ (GÜNCELLENDİ) ---
                            st.markdown("### 🎯 Güven Skoru ve Klinik Güvenilirlik Analizi")
                            
                            pred_df = pd.DataFrame({
                                "Protokol No": X_test.index,
                                "Gerçek Tanı": [class_names[i] for i in y_test],
                                "YZ Tahmini": [class_names[i] for i in y_pred],
                                "Güven Skoru (%)": confidence
                            })
                            
                            pred_df["Durum"] = np.where(pred_df["Gerçek Tanı"] == pred_df["YZ Tahmini"], "✅ Doğru", "❌ Hatalı")
                            
                            # --- İSTATİSTİK HESAPLAMA ---
                            threshold = 90.0 # %90 ve üzeri güven
                            
                            # 1. Yüksek Güvenli Hastalar
                            high_conf_df = pred_df[pred_df["Güven Skoru (%)"] >= threshold]
                            n_high = len(high_conf_df)
                            n_total_test = len(pred_df)
                            ratio_high = (n_high / n_total_test) * 100 if n_total_test > 0 else 0
                            
                            # 2. Bu gruptaki Doğruluk (Accuracy)
                            if n_high > 0:
                                n_correct_high = len(high_conf_df[high_conf_df["Durum"] == "✅ Doğru"])
                                acc_high = (n_correct_high / n_high) * 100
                            else:
                                acc_high = 0
                                
                            # 3. Kalan %20'lik dilim (Test Seti) için Genel Doğruluk
                            general_accuracy = acc * 100

                            # --- METRİKLERİ GÖSTER ---
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric(label="Toplam Test Edilen Hasta", value=n_total_test)
                            m2.metric(label="Genel Doğruluk (Tüm Test Seti)", value=f"%{general_accuracy:.2f}")
                            m3.metric(label=f"Yüksek Güvenli (>%{threshold})", value=f"{n_high} kişi", delta=f"%{ratio_high:.1f} Kapsama")
                            m4.metric(label="Yüksek Güvenli Grubun Doğruluğu", value=f"%{acc_high:.2f}", help="Modelin '%90'dan fazla eminim' dediği vakalardaki başarısı.")
                            
                            st.success(f"""
                            **📝 Makale İçin Bulgular Cümlesi:**
                            
                            "Geliştirilen yapay zeka modeli, test setindeki vakaların **%{ratio_high:.1f}**'ini (n={n_high}) **%{threshold}** ve üzeri bir güven skoru (confidence score) ile sınıflandırmıştır. 
                            Modelin kendinden emin olduğu bu yüksek güvenli grupta, tanısal doğruluk oranı (accuracy) **%{acc_high:.2f}** olarak tespit edilmiştir. Tüm test seti üzerindeki genel doğruluk oranı ise **%{general_accuracy:.2f}**'dir."
                            """)
                            
                            st.divider()
                            st.write("#### Detaylı Hasta Listesi")
                            pred_df = pred_df.sort_values("Durum", ascending=True)
                            st.dataframe(pred_df.style.apply(lambda x: ['background-color: #ffcccc' if x['Durum'] == '❌ Hatalı' else '' for i in x], axis=1), use_container_width=True)
                            
                            csv_pred = pred_df.to_csv(index=False).encode("utf-8-sig")
                            st.download_button("⬇️ Tahmin Sonuçlarını İndir (CSV)", csv_pred, "yz_tahmin_sonuclari.csv", "text/csv")

                    except Exception as e:
                        st.error(f"Hata oluştu: {e}")
                        st.info("Hata Detayı: Veri tipleri veya sütun isimleri uyumsuz olabilir.")
