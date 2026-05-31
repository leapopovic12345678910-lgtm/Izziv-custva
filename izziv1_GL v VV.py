from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"

GL_INPUT = RAW_DIR / "izziv1_GL_podatki.xlsx"
G1_INPUT = RAW_DIR / "anketa_G1_vzburjenje.xlsx"

VV_OUTPUT_XLSX = PROCESSED_DIR / "izziv1_VV_rezultati.xlsx"
VV_OUTPUT_CSV = PROCESSED_DIR / "izziv1_VV_rezultati.csv"

# =========================
# UCITAVANJE PODATAKA
# =========================

GL = pd.read_excel(GL_INPUT)
G1 = pd.read_excel(G1_INPUT)

# =========================
# PRIPREMA
# =========================

# emotion -> arousal
emotion_col = "GL čustvo" if "GL čustvo" in G1.columns else "emotion"
arousal_col = "Vzburjenje" if "Vzburjenje" in G1.columns else "arousal"
arousal_dict = dict(zip(G1[emotion_col], G1[arousal_col]))

# lista emocija
emotions = list(arousal_dict.keys())

# =========================
# FUNKCIJA ZA IZRAČUN
# =========================

def calculate_arousal(row):

    total = row[emotions].sum()

    if total == 0:
        return 0

    weighted_sum = 0

    for emotion in emotions:

        weight = row[emotion] / total

        weighted_sum += weight * arousal_dict[emotion]

    return weighted_sum

# nova VV vrednost vzburjenja
GL["VV_arousal"] = GL.apply(calculate_arousal, axis=1)

# valenca ostaje ista
GL["VV_valence"] = GL["Valenca"]

# =========================
# SHRANJEVANJE
# =========================

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
GL.to_excel(VV_OUTPUT_XLSX, index=False)
GL.to_csv(VV_OUTPUT_CSV, index=False)

print("Uspešno!")
print(f"Shranjeno: {VV_OUTPUT_XLSX}")
print(f"Shranjeno: {VV_OUTPUT_CSV}")
print(GL[["VV_valence", "VV_arousal"]].head())
