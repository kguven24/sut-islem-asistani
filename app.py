import streamlit as st
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai
from groq import Groq

st.set_page_config(
    page_title="SUT İşlem Asistanı",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 SUT İşlem Asistanı")
st.caption("Tanıya göre SGK Sağlık Uygulama Tebliği kapsamındaki uygun işlemleri bulur · Pediatri · Nefroloji · Çocuk Nefrolojisi")


# ── Data ───────────────────────────────────────────────────────────────────────

@st.cache_data
def load_procedures():
    import pathlib, urllib.request
    # Try local paths first, then fall back to GitHub raw URL
    candidates = [
        pathlib.Path(__file__).resolve().parent / "data" / "procedures.json",
        pathlib.Path("data/procedures.json"),
        pathlib.Path("/mount/src/sut-islem-asistani/data/procedures.json"),
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    # Download from GitHub if local file not found
    url = "https://raw.githubusercontent.com/kguven24/sut-islem-asistani/main/data/procedures.json"
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode("utf-8"))


SPECIALTY_KEYWORDS = {
    "Pediatri": ["PEDİYATRİ", "ÇOCUK", "YENİDOĞAN", "PEDIATR", "BEBEK", "KOT", "KUVÖZ"],
    "Nefroloji": ["NEFROLOJİ", "NEFRO", "DİYALİZ", "HEMODİYALİZ", "PERİTON", "BÖBREK", "ÜRİNER", "ÜROLOJİ", "TRANSPLANT"],
    "Çocuk Nefrolojisi": ["NEFROLOJİ", "NEFRO", "DİYALİZ", "HEMODİYALİZ", "PERİTON", "BÖBREK", "ÜRİNER", "ÜROLOJİ", "TRANSPLANT", "PEDİYATRİ", "ÇOCUK", "YENİDOĞAN", "PEDIATR", "BEBEK"],
    "Tümü": [],
}

GENERAL_KEYWORDS = [
    "HEKİM MUAYENELERİ", "MUAYENE", "RAPOR",
    "BİYOKİMYA", "HEMATOLOJİ", "MİKROBİYOLOJİ", "İMMÜNOLOJİ",
    "KAN", "İDRAR", "LAB", "TETKİK",
    "RADYOLOJİ", "ULTRASONOGRAFİ", "BT", "MR", "SİNTİGRAFİ", "GÖRÜNTÜLEME",
    "YATAK", "YOĞUN BAKIM", "ACİL", "TPN",
    "GENETİK", "PATOLOJİ", "SİTOLOJİ",
    "EKO", "EKG", "ELEKTRO",
    "ENDOSKOPİ",
    "AFEREZ", "TRANSFÜZYON", "KAN ÜRÜNLERİ",
]


def prefilter(procedures, specialty, diagnosis=""):
    spec_kw = SPECIALTY_KEYWORDS.get(specialty, [])
    # Extract meaningful words from diagnosis for relevance scoring
    diag_words = [w.upper() for w in re.split(r"[\s,/\-]+", diagnosis) if len(w) > 3]

    scored = []
    for p in procedures:
        sec = p.get("section", "").upper()
        name = p.get("name", "").upper()
        desc = p.get("description", "").upper()

        is_general = any(kw in sec or kw in name for kw in GENERAL_KEYWORDS)
        is_specialty = not spec_kw or any(kw in sec or kw in name for kw in spec_kw)

        if not (is_general or is_specialty):
            continue

        # Score by how many diagnosis words appear in procedure name/desc
        score = sum(1 for w in diag_words if w in name or w in desc or w in sec)
        scored.append((score, p))

    # Sort: diagnosis-matching first, then general
    scored.sort(key=lambda x: -x[0])
    # Cap at 150 procedures — keeps total request under 6k tokens (Groq free limit)
    return [p for _, p in scored[:150]]


def build_procedure_text(procedures):
    lines = []
    for p in procedures:
        # Omit description to save tokens — code + name is enough for matching
        lines.append(f"[{p['code']}] {p['name']}")
    return "\n".join(lines)


# ── Prompt ─────────────────────────────────────────────────────────────────────

def build_prompt(diagnosis, procedures_text, specialty):
    system = (
        f"Sen Türk sağlık sisteminde {specialty} uzmanı bir klinisyensin.\n"
        "Görevin: verilen tanı için aşağıdaki SUT işlem listesinden tıbben uygun işlemleri seçmek.\n\n"
        "KURALLAR:\n"
        "1. Sadece kanıta dayalı tıp standartlarına ve tıbbi etiğe uygun işlemleri seç.\n"
        "2. Yalnızca listede yer alan işlemleri seç — listede olmayan işlem ekleme.\n"
        "3. Her işlem için 1-2 cümle Türkçe klinik gerekçe yaz.\n"
        "4. Kontrendike veya gereksiz işlemleri dahil etme.\n"
        "5. SADECE geçerli JSON döndür, başka metin ekleme:\n"
        '   {"uygun_islemler": [{"kod": "...", "ad": "...", "gerekce": "..."}]}\n\n'
        f"SUT İŞLEM LİSTESİ:\n{procedures_text}"
    )
    user = f"Tanı: {diagnosis}\n\nBu tanı için SUT listesinden uygun işlemleri belirle."
    return system, user


# ── AI Queries ─────────────────────────────────────────────────────────────────

def _parse_json(raw):
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"uygun_islemler": []}


def query_gemini(diagnosis, procedures_text, specialty, api_key):
    genai.configure(api_key=api_key)
    system, user = build_prompt(diagnosis, procedures_text, specialty)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",  # free tier, high quota
        system_instruction=system,
    )
    response = model.generate_content(user)
    return _parse_json(response.text)


def query_groq(diagnosis, procedures_text, specialty, api_key):
    system, user = build_prompt(diagnosis, procedures_text, specialty)
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # 20k TPM free limit vs 6k for 70b
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=4096,
        temperature=0.1,
    )
    return _parse_json(response.choices[0].message.content)


def run_both(diagnosis, procedures_text, specialty, gemini_key, groq_key):
    results = {}
    fns = {
        "Gemini 2.0 Flash": (query_gemini, gemini_key),
        "LLaMA 3.3 70B (Groq)": (query_groq, groq_key),
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(fn, diagnosis, procedures_text, specialty, key): name
            for name, (fn, key) in fns.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = {"error": str(e), "uygun_islemler": []}
    return results


# ── Cross-check merge ──────────────────────────────────────────────────────────

def merge_results(results):
    by_code = {}
    for source, data in results.items():
        for item in data.get("uygun_islemler", []):
            code = item.get("kod", "").strip()
            if not code:
                continue
            if code not in by_code:
                by_code[code] = {"ad": item.get("ad", ""), "sources": {}}
            by_code[code]["sources"][source] = item.get("gerekce", "")

    agreed, single = [], []
    for code, info in by_code.items():
        entry = {"kod": code, "ad": info["ad"], "sources": info["sources"]}
        if len(info["sources"]) > 1:
            agreed.append(entry)
        else:
            single.append(entry)

    agreed.sort(key=lambda x: x["kod"])
    single.sort(key=lambda x: x["kod"])
    return agreed, single


# ── UI ─────────────────────────────────────────────────────────────────────────

procedures = load_procedures()

def get_key(env_name, secret_name):
    return os.environ.get(env_name) or st.secrets.get(secret_name, "")

gemini_key = get_key("GEMINI_API_KEY", "GEMINI_API_KEY")
groq_key   = get_key("GROQ_API_KEY",   "GROQ_API_KEY")

if not gemini_key or not groq_key:
    st.warning(
        "API anahtarları eksik. Lütfen Streamlit Secrets veya ortam değişkeni olarak "
        "`GEMINI_API_KEY` ve `GROQ_API_KEY` tanımlayın."
    )

col1, col2 = st.columns([3, 1])
with col1:
    diagnosis = st.text_input(
        "Tanı / Şikayet",
        placeholder="Örn: Kronik böbrek yetmezliği, Nefrotik sendrom, Akut glomerülonefrit…",
    )
with col2:
    specialty = st.selectbox(
        "Uzmanlık Alanı",
        ["Çocuk Nefrolojisi", "Nefroloji", "Pediatri", "Tümü"],
    )

search = st.button(
    "🔍 Uygun İşlemleri Bul",
    type="primary",
    disabled=not (diagnosis.strip() and gemini_key and groq_key),
)

if search and diagnosis.strip():
    filtered = prefilter(procedures, specialty, diagnosis.strip())
    proc_text = build_procedure_text(filtered)

    with st.spinner(f"Gemini ve LLaMA paralel taranıyor ({len(filtered)} işlem)…"):
        ai_results = run_both(diagnosis.strip(), proc_text, specialty, gemini_key, groq_key)

    for name, data in ai_results.items():
        if "error" in data:
            st.error(f"{name} hatası: {data['error']}")

    agreed, single = merge_results(ai_results)
    total = len(agreed) + len(single)

    if total == 0:
        st.info("Bu tanı için uygun işlem bulunamadı.")
    else:
        st.success(f"**{total} uygun işlem bulundu** — {len(agreed)} her iki AI tarafından onaylandı")
        st.markdown("---")

        if agreed:
            st.markdown("### ✅ Her İki AI Tarafından Önerilen İşlemler")
            st.caption("Yüksek güven — Gemini ve LLaMA 3.3 hem uygunluğu hem gerekçeyi doğruladı")
            for item in agreed:
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.code(item["kod"], language=None)
                with c2:
                    st.markdown(f"**{item['ad']}**")
                    for src, gerekce in item["sources"].items():
                        st.caption(f"*{src}:* {gerekce}")
                st.markdown("---")

        if single:
            st.markdown("### ⚠️ Tek AI Tarafından Önerilen İşlemler")
            st.caption("Gözden geçirin — yalnızca bir kaynak önerdi")
            for item in single:
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.code(item["kod"], language=None)
                with c2:
                    st.markdown(f"**{item['ad']}**")
                    for src, gerekce in item["sources"].items():
                        st.caption(f"*{src}:* {gerekce}")
                st.markdown("---")

        with st.expander("Ham AI Yanıtları"):
            for name, data in ai_results.items():
                st.markdown(f"**{name}** — {len(data.get('uygun_islemler', []))} işlem")

st.sidebar.markdown("### Hakkında")
st.sidebar.markdown(
    "SGK **Sağlık Uygulama Tebliği** (SUT) verilerine dayanır.\n\n"
    "**Kaynak:** mevzuat.gov.tr · Ocak 2025 güncel SUT\n\n"
    "**AI Modeller:** Gemini 2.0 Flash · LLaMA 3.3 70B\n\n"
    "⚠️ Bu araç karar desteği amaçlıdır; klinik değerlendirmenin yerini tutmaz."
)
st.sidebar.markdown(f"**Toplam SUT işlemi:** {len(procedures):,}")
