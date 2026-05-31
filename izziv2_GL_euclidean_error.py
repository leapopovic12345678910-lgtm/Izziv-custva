from pathlib import Path
import csv
import math


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"

GL_INPUT = RAW_DIR / "izziv2_GL_all.csv"
OUTPUT = PROCESSED_DIR / "izziv2_GL_euclidean_error.csv"

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

MAX_EMOTION_VALUE = 8.0
MAX_DISTANCE = MAX_EMOTION_VALUE * math.sqrt(len(EMOTIONS))


def clean_row(row):
    return {key.strip(): (value or "").strip() for key, value in row.items()}


def target_value(clip, emotion):
    return MAX_EMOTION_VALUE if emotion == GROUND_TRUTH_BY_CLIP[clip] else 0.0


def calculate_euclidean_error(row):
    squared_distance = 0.0
    clip = row["klip"]

    for emotion in EMOTIONS:
        actual = float(row[emotion])
        target = target_value(clip, emotion)
        squared_distance += (actual - target) ** 2

    distance = math.sqrt(squared_distance)
    return distance / MAX_DISTANCE


def main():
    output_rows = []

    with GL_INPUT.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            row = clean_row(raw_row)
            error = calculate_euclidean_error(row)
            output_rows.append(
                {
                    "klip": row["klip"],
                    "original method": row["original method"],
                    "šifra": row["šifra"],
                    "euclidean_error": f"{error:.6f}".rstrip("0").rstrip("."),
                }
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["klip", "original method", "šifra", "euclidean_error"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
