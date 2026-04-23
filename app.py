import streamlit as st
import json
import os
import re
import anthropic

st.set_page_config(
    page_title="SUT İşlem Asistanı",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 SUT İşlem Asistanı")
st.caption("Tanıya göre SGK Sağlık Uygulama Tebliği kapsamındaki uygun işlemleri bulur · Pediatri, Nefroloji, Çocuk Nefrolojisi")


@st.cache_data
def load_procedures():
    path = os.path.join(os.path.dirname(__file__), "data", "procedures.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SPECIALTY_KEYWORDS = {
    "Pediatri": [
        "PEDİYATRİ", "ÇOCUK", "YENİDOĞAN", "PEDIATR", "BEBEK", "KOT", "KUVÖZ"
    ],
    "Nefroloji": [
        "NEFROLOJİ", "NEFRO", "DİYALİZ", "HEMODİYALİZ", "PERİTON",
        "BÖBREK", "ÜRİNER", "ÜROLOJİ", "TRANSPLANT"
    ],
    "Çocuk Nefrolojisi": [
        "NEFROLOJİ", "NEFRO", "DİYALİZ", "HEMODİYALİZ", "PERİTON",
        "BÖBREK", "ÜRİNER", "ÜROLOJİ", "TRANSPLANT",
        "PEDİYATRİ", "ÇOCUK", "YENİDOĞAN", "PEDIATR", "BEBEK"
    ],
    "Tümü": [],
}

GENERAL_KEYWORDS = [
    "HEKİM MUAYENELERİ", "MUAYENE", "RAPOR",
    "BİYOKİMYA", "HEMATOLOJİ", "MİKROBİYOLOJİ", "İMMÜNOLOJİ",
    "KAN", "İDRAR", "LAB", "TETKİK",
    "RADYOLOJİ", "ULTRASONOGRAFİ", "BT", "MR", "SİNTİGRAFİ", "GÖRÜNTÜLEME",
    "YATAK", "YOĞUN BAKIM", "ACİL", "TPN",
    "GENETİK", "PAT", "ATOLOJI", "SİTOLOJİ",
    "EKO", "EKG", "ELEKTRO",
    "ENDOSKOPİ",
    "AFEREZ", "TRANSFÜZYON", "KAN ÜRÜNLERİ",
]


def prefilter(procedures, specialty):
    spec_kw = SPECIALTY_KEYWORDS.get(specialty, [])
    result = []
    for p in procedures:
        sec = p.get("section", "").upper()
        name = p.get("name", "").upper()
        is_general = any(kw in sec or kw in name for kw in GENERAL_KEYWORDS)
        is_specialty = not spec_kw or any(kw in sec or kw in name for kw in spec_kw)
        if is_general or is_specialty:
            result.append(p)
    return result


def build_procedure_text(procedures):
    lines = []
    for p in procedures:
        line = f"[{p['code']}] {p['name']}"
        if p.get("description"):
            short_desc = p["description"][:120]
            line += f" — {short_desc}"
        lines.append(line)
    return "\n".join(lines)


def query(diagnosis, procedures_text, specialty):
    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("ANTHROPIC_API_KEY bulunamadı. Lütfen ortam değişkeni veya Streamlit secret olarak tanımlayın.")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    system_content = [
        {
            "type": "text",
            "text": f"""Sen Türk sağlık sisteminde {specialty} uzmanı bir klinisyensin.
Görevin: verilen tanı/şikayet için, aşağıdaki SGK Sağlık Uygulama Tebliği (SUT) işlem listesinden tıbben uygun işlemleri seçmek.

KURALLAR:
1. Sadece tıbbi etik, kanıta dayalı tıp standartları ve SUT mevzuatına uygun işlemleri seç.
2. Yalnızca aşağıdaki listede yer alan işlemleri seç — listede olmayan işlem ekleme.
3. Her işlem için kısa (1-2 cümle) Türkçe klinik gerekçe yaz.
4. Kontrendike, gereksiz veya başka bir uzmanlık alanına ait işlemleri dahil etme.
5. Yanıtını SADECE geçerli JSON olarak döndür, başka metin ekleme:
   {{"uygun_islemler": [{{"kod": "...", "ad": "...", "gerekce": "..."}}]}}

SUT İŞLEM LİSTESİ:""",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": procedures_text,
            "cache_control": {"type": "ephemeral"},
        },
    ]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system_content,
        messages=[
            {
                "role": "user",
                "content": f"Tanı / Şikayet: **{diagnosis}**\n\nBu tanı için SUT listesinden uygun işlemleri belirle.",
            }
        ],
    )

    raw = response.content[0].text.strip()
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"uygun_islemler": []}


# ── UI ─────────────────────────────────────────────────────────────────────────

procedures = load_procedures()

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

search = st.button("🔍 Uygun İşlemleri Bul", type="primary", disabled=not diagnosis.strip())

if search and diagnosis.strip():
    filtered = prefilter(procedures, specialty)
    proc_text = build_procedure_text(filtered)

    with st.spinner(f"'{diagnosis}' için {len(filtered)} işlem arasında taranıyor…"):
        result = query(diagnosis.strip(), proc_text, specialty)

    if result:
        items = result.get("uygun_islemler", [])
        if not items:
            st.info("Bu tanı için listede uygun bir işlem bulunamadı.")
        else:
            st.success(f"**{len(items)} uygun işlem bulundu**")
            st.markdown("---")
            for item in items:
                with st.container():
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        st.code(item.get("kod", "—"), language=None)
                    with c2:
                        st.markdown(f"**{item.get('ad', '')}**")
                        st.caption(item.get("gerekce", ""))
                st.markdown("---")

st.sidebar.markdown("### Hakkında")
st.sidebar.markdown(
    "SGK **Sağlık Uygulama Tebliği** (SUT) verilerine dayanır.\n\n"
    "**Kaynak:** mevzuat.gov.tr · Ocak 2025 güncel SUT\n\n"
    "⚠️ Bu araç karar desteği amaçlıdır; klinik değerlendirmenin yerini tutmaz."
)
st.sidebar.markdown(f"**Toplam SUT işlemi:** {len(procedures):,}")
