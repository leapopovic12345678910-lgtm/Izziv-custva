import pandas as pd

# =========================
# UCITAVANJE PODATAKA
# =========================

GL = pd.read_excel("GL_podatki.xlsx")
print(GL.columns)
G1 = pd.read_excel("G1_vzburjenje.xlsx")

# =========================
# PRIPREMA
# =========================

# emotion -> arousal
arousal_dict = dict(zip(G1["emotion"], G1["arousal"]))

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

GL.to_excel("VV_rezultati.xlsx", index=False)

print("Uspešno!")
print(GL[["VV_valence", "VV_arousal"]].head())