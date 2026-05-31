from pathlib import Path
import csv
import math


BASE_DIR = Path(__file__).resolve().parent
INTERIM_DIR = BASE_DIR / "data/interim"
PROCESSED_DIR = BASE_DIR / "data/processed"

GL_INPUT = INTERIM_DIR / "izziv2_GL_all_normalized.csv"
G1_INPUT = INTERIM_DIR / "anketa_G1_vzburjenje_normalized.csv"
OUTPUT = PROCESSED_DIR / "izziv2_GL_weighted_rmse_error.csv"
ERROR_METHOD = "weighted_rmse"

EMOTIONS = [
    "amusement",
    "anger",
    "anxiety",
    "confusion",
    "contempt",
    "disgust",
    "embarrassment",
    "fear",
    "guilt",
    "happiness",
    "interest",
    "joy",
    "love",
    "pride",
    "sadness",
    "shame",
    "surprise",
    "unhappiness",
]

GROUND_TRUTH_BY_CLIP = {
    "ples": "happiness",
    "hodnik": "fear",
    "restavracija": "amusement",
}


def clean_row(row):
    return {key.strip(): (value or "").strip() for key, value in row.items()}


def load_weights():
    """Use absolute valence distance from neutral, then normalize to sum to 1."""
    distances = {}

    with G1_INPUT.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            row = clean_row(raw_row)
            emotion = row["GL čustvo"]
            if emotion not in EMOTIONS:
                continue
            distances[emotion] = abs(float(row["Valenca"]) - 0.5)

    missing = [emotion for emotion in EMOTIONS if emotion not in distances]
    if missing:
        raise ValueError(f"Missing G1 weights for emotions: {', '.join(missing)}")

    total = sum(distances.values())
    if total == 0:
        raise ValueError("All emotion weights are zero; cannot normalize weights.")

    return {emotion: distance / total for emotion, distance in distances.items()}


def target_value(clip, emotion):
    return 1.0 if emotion == GROUND_TRUTH_BY_CLIP[clip] else 0.0


def calculate_weighted_rmse(row, weights):
    weighted_squared_error = 0.0
    clip = row["klip"]

    for emotion in EMOTIONS:
        actual = float(row[emotion])
        target = target_value(clip, emotion)
        weighted_squared_error += weights[emotion] * (actual - target) ** 2

    return math.sqrt(weighted_squared_error)


def main():
    weights = load_weights()
    output_rows = []

    with GL_INPUT.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            row = clean_row(raw_row)
            error = calculate_weighted_rmse(row, weights)
            output_rows.append(
                {
                    "klip": row["klip"],
                    "original method": row["original method"],
                    "šifra": row["šifra"],
                    "error_method": ERROR_METHOD,
                    "emotion_error": f"{error:.6f}".rstrip("0").rstrip("."),
                }
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "klip",
                "original method",
                "šifra",
                "error_method",
                "emotion_error",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
