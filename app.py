# app.py
# -*- coding: utf-8 -*-
"""
🧪 Tetkik Analiz Arayüzü — Çoklu Dosya (Optimize, Revize)
- Çoklu dosya hızlı okuma (paralel + cache)
- Bellek optimizasyonu (downcast, categorical)
- Hızlı grafikler (matplotlib; renk sabitlenmez)
- Kategorik normalizasyon:
    • Kan Grubu: A/B/AB/O + Rh(+/-)
    • Anormal Hb: HbS/HbC/HbD/HbE/HbA2↑/HbF↑/Normal
- Varyant etiketi: S/, C/, D/, E/, A2/, F/ + yüzdeli kolonlar
"""

import io, re, math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="Tetkik Analiz — Optimize", layout="wide")

REQ_COLS = ["PROTOKOL_NO", "TCKIMLIK_NO", "TETKIK_ISMI", "TEST_DEGERI", "CINSIYET"]
CATEGORICAL_TESTS = {"Kan Grubu/", "Anormal Hb/"}

# Eşikler
THRESHOLDS = {
    "HbA2 (%)": (">=", 3.5),
    "A2/":      (">=", 3.5),
    "HbC (%)":  (">",  0.0),
    "C/":       (">",  0.0),
    "HbD (%)":  (">",  0.0),
    "D/":       (">",  0.0),
    "HbE (%)":  (">",  0.0),
    "E/":       (">",  0.0),
    "HbF (%)":  (">",  2.0),
    "F/":       (">",  2.0),
    "HbS (%)":  (">",  0.0),
    "S/":       (">",  0.0),
}
GT_ZERO_DEFAULT = {"C/","D/","E/","S/","HbS (%)","HbC (%)","HbD (%)","HbE (%)","HbF (%)"}

# Kısa ad + yüzdeli sütun adlarını aynı etikete bağla
VARIANT_NUMERIC_TESTS = {
    "C/": "HbC", "HbC (%)": "HbC",
    "D/": "HbD", "HbD (%)": "HbD",
    "E/": "HbE", "HbE (%)": "HbE",
    "S/": "HbS", "HbS (%)": "HbS",
    # A2 ve F yüzdeleri eşikle yakalanacak; yine de pozitiflik amaçlı eşlemede tutalım
    "A2/": "HbA2↑", "HbA2 (%)": "HbA2↑",
    "F/":  "HbF↑",  "HbF (%)":  "HbF↑",
}

DISPLAY_LIMIT = 400
MALE_TOKENS   = {"e","erkek","m","male","bay"}
FEMALE_TOKENS = {"k","kadın","kadin","f","female","bayan"}

try:
    import polars as pl
    HAS_POLARS = True
except Exception:
    HAS_POLARS = False

# ============== Yardımcılar ============== #
_num_pat = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

def parse_numeric_cell(x) -> float | None:
    """'5,2 %', ' >2.0', '<0,1', '≈ 3,50' gibi değerlerden ilk sayıyı güvenli çeker."""
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    m = _num_pat.search(s.replace("‰", "").replace("%", ""))
    if not m:
        return None
    val = m.group(0).replace(",", ".")
    try:
        return float(val)
    except Exception:
        return None

def add_numeric_copy(frame, src_col="TEST_DEGERI", out_col="__VAL_NUM__"):
    if out_col not in frame.columns:
        frame[out_col] = frame[src_col].map(parse_numeric_cell)
    return frame

def check_columns(df: pd.DataFrame):
    return [c for c in REQ_COLS if c not in df.columns]

def normalize_sex_label(value):
    if not isinstance(value, str): return None
    low = value.strip().lower()
    if not low: return None
    if low in MALE_TOKENS: return "Erkek"
    if low in FEMALE_TOKENS: return "Kadın"
    return value

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
    row_counts = s_rows.value_counts(dropna=False).rename_axis("CINSIYET").to_frame("Satır Sayısı")
    with_id = tmp[tmp["TCKIMLIK_NO"].notna()].copy()
    if not with_id.empty:
        w = with_id.copy()
        w["__SEX_CANON__"] = w["__SEX_CANON__"].astype(object)
        patient_gender = (
            w.groupby("TCKIMLIK_NO")["__SEX_CANON__"]
             .apply(lambda s: _resolve_patient_sex(pd.Series(pd.unique(s.dropna()))))
             .reset_index(name="__SEX_RESOLVED__")
        )
        patient_counts = patient_gender["__SEX_RESOLVED__"].fillna("Bilinmiyor").value_counts(dropna=False)\
                          .rename_axis("CINSIYET").to_frame("Hasta (Benzersiz)")
    else:
        patient_counts = pd.DataFrame(columns=["Hasta (Benzersiz)"])
    summary = row_counts.join(patient_counts, how="outer").fillna(0)
    summary["Satır Sayısı"] = summary["Satır Sayısı"].astype(int)
    if "Hasta (Benzersiz)" in summary.columns:
        summary["Hasta (Benzersiz)"] = summary["Hasta (Benzersiz)"].astype(int)
    else:
        summary["Hasta (Benzersiz)"] = 0
    total_rows = int(summary["Satır Sayısı"].sum()) or 0
    total_patients = int(summary["Hasta (Benzersiz)"].sum()) or 0
    summary["% Satır"]  = (summary["Satır Sayısı"] / total_rows * 100).round(2) if total_rows else np.nan
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
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return {"count":0,"mean":np.nan,"std":np.nan,"min":np.nan,"q1":np.nan,"median":np.nan,"q3":np.nan,"max":np.nan,"cv%":np.nan,"iqr":np.nan}
    q = np.percentile(x, [25, 50, 75])
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    cv  = (std/mean*100) if mean!=0 else np.nan
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
        g0, g1 = list(unique_groups)
        x = pd.to_numeric(df[df[grp_col] == g0][val_col], errors="coerce").dropna()
        y = pd.to_numeric(df[df[grp_col] == g1][val_col], errors="coerce").dropna()
        if len(x) >= 1 and len(y) >= 1:
            stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            return f"Mann–Whitney U: U={stat:.2f}, p={p:.4g} ({g0} vs {g1})", ("MWU", stat, p, g0, g1)
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

# ======== Kategorik normalizasyon ======== #
def normalize_blood_group(x: str | None):
    if not isinstance(x, str): return None
    u = x.strip().upper().replace("İ", "I")
    if not u: return None
    abo = None
    if re.search(r"\bAB\b", u): abo = "AB"
    elif re.search(r"\bA\b", u): abo = "A"
    elif re.search(r"\bB\b", u): abo = "B"
    elif re.search(r"\bO\b|\b0\b", u): abo = "O"
    rh = None
    if re.search(r"\+|\bPOS(ITIVE)?\b|\bPOZ(ITIF)?\b|\bRH\+\b", u): rh = "Rh(+)"
    elif re.search(r"-|\bNEG(ATIVE)?\b|\bRH-\b", u): rh = "Rh(-)"
    if abo is None and rh is None: return None
    return f"{abo or ''} {rh or ''}".strip()

def norm_anormal_hb_text(x: str | None):
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

# ============== UI ============== #
st.title("⚡ Tetkik Analiz — Çoklu Dosya (Optimize, Revize)")
st.caption("Kısa adlar: Kan Grubu/, S/, Talasemi(HPLC) (A0)/, A2/, C/, D/, E/, F/ desteklenir.")

uploads = st.file_uploader("Excel dosyaları (.xlsx, .xls) — Çoklu seçim", type=["xlsx", "xls"], accept_multiple_files=True)

use_polars = st.checkbox("Polars hızlandırmayı dene (kuruluysa)", value=('pl' in globals() and HAS_POLARS))

if not uploads:
    st.info("Birden çok dosyayı aynı anda seçin.")
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

st.markdown("### 🧾 Veri Filtre Ayarları")
include_99 = st.checkbox("99 ile başlayan TCKN'leri dahil et", value=False)

work = df.copy()
if not include_99 and "TCKIMLIK_NO" in work.columns:
    work = work[~work["TCKIMLIK_NO"].astype(str).str.startswith("99")]
if chosen_sex:
    work = work[work["CINSIYET"].astype(str).isin(chosen_sex)]
if chosen_files:
    work = work[work["SOURCE_FILE"].astype(str).isin(chosen_files)]
if selected_tests:
    work = work[work["TETKIK_ISMI"].astype(str).isin(selected_tests)]

# Sayısal kopya (sağlam ayrıştırma ile)
work = add_numeric_copy(work)

# ================= VARYANT ÖZETİ ================= #
A2_KEYS = {"A2/","HbA2","HbA2 (%)","Hb A2","Hb A2 (%)"}
F_KEYS  = {"F/","HbF","HbF (%)","Hb F","Hb F (%)"}

def pick_variant_tag(g: pd.DataFrame) -> str | None:
    g = g.copy()
    g = add_numeric_copy(g)
    names = g["TETKIK_ISMI"].astype(str).str.strip()
    vals  = g["__VAL_NUM__"]

    tags = []

    # 1) Anormal Hb/ metni
    mask_txt = names.eq("Anormal Hb/")
    if mask_txt.any():
        for v in g.loc[mask_txt, "TEST_DEGERI"].dropna().astype(str):
            t = norm_anormal_hb_text(v)
            if t: tags.append(t)

    # 2) A2/F eşikleri
    mask_a2 = names.isin(A2_KEYS)
    if mask_a2.any():
        a2 = g.loc[mask_a2, "__VAL_NUM__"].dropna()
        if not a2.empty and a2.max() >= 3.5:
            tags.append("HbA2↑")

    mask_f = names.isin(F_KEYS)
    if mask_f.any():
        f = g.loc[mask_f, "__VAL_NUM__"].dropna()
        if not f.empty and f.max() > 2.0:
            tags.append("HbF↑")

    # 3) Kısa adlar + yüzdeli kolonlar (>0 ise)
    for key, label in VARIANT_NUMERIC_TESTS.items():
        m = names.eq(key)
        if m.any():
            vv = g.loc[m, "__VAL_NUM__"].dropna()
            if not vv.empty and (vv > 0).any():
                tags.append(label)

    if not tags:
        return None

    # Öncelik sırası
    for p in ["Hb S-β-thal","HbS","HbC","HbD","HbE","HbA2↑","HbF↑","Normal"]:
        if p in tags:
            return p
    return sorted(set(tags))[0]

if "VARIANT_TAG" not in work.columns:
    var_map = (work.groupby("PROTOKOL_NO", group_keys=False)
                  .apply(lambda g: pd.Series({"VARIANT_TAG": pick_variant_tag(g)}))
                  .reset_index())
    work = work.merge(var_map, on="PROTOKOL_NO", how="left")

st.header("📋 Varyant Özeti — erişkin eşikleri ile")
present = [t for t in ["Hb S-β-thal","HbS","HbC","HbD","HbE","HbA2↑","HbF↑","Normal"]
           if t in set(work["VARIANT_TAG"].dropna())]
variant_choice = st.selectbox("Varyant seç:", ["(Tümü)"] + present, index=0)

base_v = work if variant_choice == "(Tümü)" else work[work["VARIANT_TAG"] == variant_choice]

if variant_choice == "(Tümü)":
    freq = work["VARIANT_TAG"].value_counts(dropna=True).rename_axis("Varyant").to_frame("N").reset_index()
    total = int(freq["N"].sum()) if not freq.empty else 0
    if total: freq["%"] = (freq["N"]/total*100).round(2)
    st.subheader("Varyant Frekansları")
    st.dataframe(freq, use_container_width=True)
    st.download_button("⬇️ Varyant frekansları (CSV)",
                       data=freq.to_csv(index=False).encode("utf-8-sig"),
                       file_name="varyant_frekans.csv", mime="text/csv")

def _mean_sd(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return "—" if s.empty else f"{s.mean():.2f} ± {s.std(ddof=1):.2f}"

PARAMS = {
    "HbA":           ("HbA (%)",     "94–98"),
    "HbA2 (%)":      ("HbA₂ (%)",    "2–3.5"),
    "A2/":           ("HbA₂ (%)",    "2–3.5"),
    "HbD (%)":       ("Hb D (%)",    "0–100"),
    "HbC (%)":       ("Hb C (%)",    "0–100"),
    "HbE (%)":       ("Hb E (%)",    "0–100"),
    "HbF (%)":       ("Hb F (%)",    "0–2"),
    "F/":            ("Hb F (%)",    "0–2"),
}

table_fm = pd.DataFrame()
if variant_choice != "(Tümü)":
    rows = []
    for tetkik_key, (disp, ref) in PARAMS.items():
        subp = base_v[base_v["TETKIK_ISMI"].astype(str).str.strip() == tetkik_key].copy()
        if subp.empty: 
            continue
        subp = add_numeric_copy(subp)
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
    sub = work[work["TETKIK_ISMI"].astype(str).str.strip() == test_name].copy()
    if sub.empty:
        st.warning(f"{test_name} verisi bulunamadı.")
        continue

    st.subheader(f"🔍 {test_name}")

    raw_text = sub["TEST_DEGERI"].astype(str).str.strip()
    normalized = raw_text.map(normalize_blood_group) if test_name == "Kan Grubu/" else raw_text.map(norm_anormal_hb_text)

    if test_name == "Anormal Hb/":
        sub_nonempty = sub[raw_text.ne("") & sub["TEST_DEGERI"].notna()].copy()
        if sub_nonempty.empty:
            st.info("Anormal Hb/ için dolu metin bulunamadı.")
        else:
            map_tc = (
                sub_nonempty.assign(_val=raw_text.loc[sub_nonempty.index])
                .groupby("_val", dropna=False)["TCKIMLIK_NO"]
                .apply(lambda s: ", ".join(sorted({str(x) for x in s.dropna().astype(str)})) or "—")
                .reset_index().rename(columns={"_val": "Ham Değer", "TCKIMLIK_NO": "TCKIMLIK_NO (liste)"})
            )
            st.markdown("**Ham yazımlar (TC listeli)**")
            st.dataframe(map_tc, use_container_width=True)
            st.download_button("⬇️ AnormalHb_ham_yazim_TC_listesi.csv",
                               data=map_tc.to_csv(index=False).encode("utf-8-sig"),
                               file_name="AnormalHb_ham_yazim_TC_listesi.csv", mime="text/csv")

        edit_cols = [c for c in ["PROTOKOL_NO","TCKIMLIK_NO","CINSIYET","SOURCE_FILE","TEST_DEGERI"] if c in sub_nonempty.columns]
        edit_df = sub_nonempty[edit_cols].copy()
        clean_col = "ANORMAL_HB_CLEAN"
        edit_df[clean_col] = (sub_nonempty[clean_col] if clean_col in sub_nonempty.columns
                              else normalized.loc[sub_nonempty.index].fillna("").astype(str))
        st.markdown("**Düzenlenebilir tablo (CLEAN değerini yazın)**")
        edited = st.data_editor(edit_df, use_container_width=True, key="anormalhb_editor",
                                column_config={"TEST_DEGERI": st.column_config.TextColumn(label="ORIGINAL", disabled=True),
                                               clean_col: st.column_config.TextColumn(label="CLEAN (düzenlenebilir)")})
        col_apply, col_over = st.columns([1,1])
        with col_apply:
            apply_now = st.button("✅ Uygula ve kaydet (oturum içi)", key="apply_anormalhb")
        with col_over:
            overwrite_main = st.checkbox("ORIGINAL sütununu da CLEAN ile değiştir", value=False, key="over_anormalhb")
        if apply_now and not edited.empty:
            upd = edited[[c for c in ["PROTOKOL_NO","TEST_DEGERI",clean_col] if c in edited.columns]].copy()
            upd.rename(columns={clean_col: "__CLEAN_TMP__"}, inplace=True)
            key_proto = work["PROTOKOL_NO"].astype(str) if "PROTOKOL_NO" in work.columns else pd.Series("", index=work.index)
            key_test  = work["TEST_DEGERI"].astype(str).str.strip()
            for _, r in upd.iterrows():
                proto = str(r.get("PROTOKOL_NO","")); orig  = str(r.get("TEST_DEGERI","")).strip()
                mask = (key_proto == proto) & (key_test == orig)
                work.loc[mask, clean_col] = r["__CLEAN_TMP__"]
                if overwrite_main:
                    work.loc[mask, "TEST_DEGERI"] = r["__CLEAN_TMP__"]
            st.success("Anormal Hb/ CLEAN değerleri uygulandı.")
            st.download_button("⬇️ Güncellenmiş veri (CSV)",
                               data=work.to_csv(index=False).encode("utf-8-sig"),
                               file_name="guncellenmis_veri.csv", mime="text/csv")
        continue

    # Kan Grubu/
    sub_text = raw_text[raw_text.str.contains(r"[A-Za-zİıÖöÜüÇçŞş]", na=False)]
    value_counts = (sub_text.value_counts(dropna=False).rename_axis("Benzersiz Değer").reset_index(name="Frekans")
                    if not sub_text.empty else pd.DataFrame(columns=["Benzersiz Değer","Frekans"]))
    st.markdown("**Ham Yazımlar**")
    st.dataframe(value_counts, use_container_width=True)
    st.download_button(f"⬇️ {test_name.strip('/')}_benzersiz_degerler.csv",
                       data=value_counts.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{test_name.strip('/')}_benzersiz_degerler.csv", mime="text/csv")

    norm_counts = normalized.value_counts(dropna=False).rename_axis("Kategori (normalize)").reset_index(name="N")
    if not norm_counts.empty:
        totalN = int(norm_counts["N"].sum()); norm_counts["%"] = (norm_counts["N"]/totalN*100).round(2)
    else:
        norm_counts = pd.DataFrame(columns=["Kategori (normalize)","N","%"])
    st.markdown("**Normalize Edilmiş Kategoriler**")
    st.dataframe(norm_counts, use_container_width=True)
    st.download_button(f"⬇️ {test_name.strip('/')}_normalize_frekans.csv",
                       data=norm_counts.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{test_name.strip('/')}_normalize_frekans.csv", mime="text/csv")

    cat_name = "__CAT__"
    sub = sub.assign(**{cat_name: normalized})
    freq_all = sub[cat_name].value_counts(dropna=False).rename_axis("Kategori").to_frame("N").reset_index()
    totalN = int(freq_all["N"].sum()) if not freq_all.empty else 0
    if totalN: freq_all["%"] = (freq_all["N"]/totalN*100).round(2)
    freq_by_sex = (sub.pivot_table(index=cat_name, columns="CINSIYET", values="PROTOKOL_NO",
                                   aggfunc="count", fill_value=0).astype(int)
                   .reset_index().rename(columns={cat_name:"Kategori"}))
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

# ================= Tetkik Bazlı Analiz (Seçim) ================= #
st.header("📊 Tetkik Bazlı Analiz (Seçim)")
results_rows = []
overall_pool = []

for test_name in selected_tests:
    if test_name in CATEGORICAL_TESTS:
        continue

    sub = work[work["TETKIK_ISMI"].astype(str).str.strip() == test_name].copy()
    if sub.empty: 
        continue

    use_threshold = st.checkbox(f"‘{test_name}’ için erişkin eşiğini uygula",
                                value=(test_name in THRESHOLDS), key=f"th_{test_name}")
    use_gt_zero  = st.checkbox(f"‘{test_name}’ için sadece > 0 değerleri dahil et",
                                value=(test_name in GT_ZERO_DEFAULT), key=f"gt0_{test_name}")

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
    by_sex  = (sub_work.groupby("CINSIYET", dropna=False)["__VAL_NUM__"]
               .agg(count="count", mean="mean", std="std", min="min", median="median", max="max")).reset_index()
    by_file = (sub_work.groupby("SOURCE_FILE", dropna=False)["__VAL_NUM__"]
               .agg(count="count", mean="mean", std="std", min="min", median="median", max="max")).reset_index()
    _msg_df = sub_work.rename(columns={"__VAL_NUM__": "VAL"})
    msg, _ = nonparametric_test_by_group(_msg_df, "VAL", "CINSIYET")

    overall_pool.extend(pd.to_numeric(_msg_df["VAL"], errors="coerce").dropna().tolist())

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

    pos_cols = [c for c in ["PROTOKOL_NO", "TCKIMLIK_NO", "CINSIYET", "SOURCE_FILE"] if c in sub_work.columns]
    pos_tbl = sub_work[pos_cols + ["__VAL_NUM__"]].sort_values("__VAL_NUM__", ascending=False)
    st.write("**Filtre sonrası kayıtlar**")
    st.dataframe(pos_tbl, use_container_width=True)
    st.download_button("⬇️ TCKIMLIK_NO listesi (CSV)",
                       data=pos_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{test_name}_filtre_sonrasi.csv", mime="text/csv")

if results_rows:
    st.header("🧾 Toplu Özet Tablosu (Seçili Tetkikler)")
    res_df = pd.DataFrame(results_rows)
    if len(overall_pool) > 0:
        overall_stats = descr_stats_fast(pd.Series(overall_pool))
        overall_row = {
            "TETKIK_ISMI": "GENEL TOPLAM",
            "N": overall_stats["count"], "Mean": overall_stats["mean"],
            "Median": overall_stats["median"], "Std": overall_stats["std"],
            "Min": overall_stats["min"], "Q1": overall_stats["q1"],
            "Q3": overall_stats["q3"], "Max": overall_stats["max"],
            "Normalite": "—", "Test": "—",
        }
        res_df = pd.concat([res_df, pd.DataFrame([overall_row])], ignore_index=True)
    st.dataframe(res_df, use_container_width=True)
    export_df(res_df, name="tetkik_ozet.csv")

st.caption("Not: Kısa anahtarlar (S/, C/, D/, E/, A2/, F/) ve hemogram dışı diğer test isimleri desteklenir.")
