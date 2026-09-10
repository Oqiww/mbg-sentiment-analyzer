"""
preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Faithful replication of the Stream B preprocessing pipeline as used during
model training in:
    Syauqi Gathan Setyapratama_2617_Week4BD_LAS26_Notebook.ipynb

Stream B Pipeline:
  1. unicodedata.normalize("NFKC", text)
  2. Emoji to Indonesian semantic tag replacement (EMOJI_INDO_MAP)
  3. text.lower()
  4. Remove URLs  (https?://S+ | www.S+)
  5. Remove @mentions
  6. Remove #hashtags
  7. Slang dictionary replacement (word-level, using CURATED_SLANG_DICT)
  8. Collapse repeated chars:  re.sub(r"(.)\1{2,}", r"\1\1", text)
  9. Collapse extra whitespace

DO NOT modify this pipeline - it must match training exactly.
"""

import re
import unicodedata
from typing import Dict


# Emoji to Indonesian semantic tag mapping
# Exactly as defined in the training notebook (Section 6)
EMOJI_INDO_MAP: Dict[str, str] = {
    "😂": " emoji_tertawa ",
    "🤣": " emoji_tertawa ",
    "😆": " emoji_tertawa ",
    "😹": " emoji_tertawa ",
    "😄": " emoji_tertawa ",
    "😀": " emoji_tertawa ",
    "😃": " emoji_tertawa ",
    "😭": " emoji_tangis_sedih ",
    "😢": " emoji_tangis_sedih ",
    "🥺": " emoji_memelas_sedih ",
    "😔": " emoji_sedih ",
    "💔": " emoji_patah_hati ",
    "🥀": " emoji_layu_sedih ",
    "🥲": " emoji_senyum_sedih ",
    "😡": " emoji_marah ",
    "🤬": " emoji_marah_kasar ",
    "😤": " emoji_kesal_marah ",
    "😾": " emoji_marah ",
    "🤮": " emoji_jijik_mual ",
    "💩": " emoji_kotoran_buruk ",
    "👍": " emoji_jempol_setuju ",
    "👏": " emoji_tepuk_tangan ",
    "💯": " emoji_sempurna_mantap ",
    "👌": " emoji_bagus_setuju ",
    "🔥": " emoji_hebat_semangat ",
    "🎉": " emoji_perayaan_selamat ",
    "🙏": " emoji_doa_syukur ",
    "🤲": " emoji_berdoa ",
    "🤝": " emoji_sepakat_kerja_sama ",
    "🥰": " emoji_cinta_kasih ",
    "❤": " emoji_cinta ",
    "💖": " emoji_cinta ",
    "😍": " emoji_suka_kagum ",
    "🫶": " emoji_cinta_hati ",
    "🫰": " emoji_cinta_korea ",
    "😁": " emoji_senyum_lebar ",
    "😊": " emoji_senyum_ramah ",
    "🙂": " emoji_senyum ",
    "☺": " emoji_senyum ",
    "😌": " emoji_lega ",
    "😇": " emoji_malaikat_baik ",
    "🗿": " emoji_sarkas_batu ",
    "😏": " emoji_sarkas_sinis ",
    "🙄": " emoji_meremehkan_kesal ",
    "🤡": " emoji_badut_konyol ",
    "🥱": " emoji_bosan_menguap ",
    "🤔": " emoji_berpikir_tanya ",
    "🧐": " emoji_menyelidiki ",
    "👀": " emoji_melihat_penasaran ",
    "😱": " emoji_kaget_panik ",
    "🤯": " emoji_tercengang ",
    "😮": " emoji_terkejut ",
    "😳": " emoji_malu_kaget ",
    "😬": " emoji_meringis_takut ",
    "🤦": " emoji_tepuk_jidat_kecewa ",
    "🤷": " emoji_tidak_tahu_pasrah ",
    "🫠": " emoji_meleleh_pasrah ",
    "🍗": " emoji_makanan_ayam ",
    "🥚": " emoji_makanan_telur ",
    "⚠️": " emoji_peringatan_bahaya ",
}


def _base_clean(text: str, emoji_map: Dict[str, str]) -> str:
    """
    Base cleaning shared by Stream B.
    Steps (from notebook):
      1. NFKC unicode normalization
      2. Emoji replacement
      3. lowercase
      4. Remove URLs
      5. Remove @mentions
      6. Remove #hashtags
    """
    text = unicodedata.normalize("NFKC", text)
    for em, tag in emoji_map.items():
        if em in text:
            text = text.replace(em, tag)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    return text


def clean_stream_b(text: str, slang_dict: Dict[str, str]) -> str:
    """
    Stream B preprocessing - the pipeline used for IndoBERT-base-p1.

    Steps (from notebook):
      1. _base_clean  (NFKC > emoji replace > lower > remove URLs/@/#)
      2. Word-level slang replacement using CURATED_SLANG_DICT
      3. Collapse repeated chars (3+) to max 2
      4. Collapse extra whitespace + strip
    """
    if not isinstance(text, str):
        return ""
    text = _base_clean(text, EMOJI_INDO_MAP)
    words = text.split()
    norm_words = [slang_dict.get(w, w) for w in words]
    text = " ".join(norm_words)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()
