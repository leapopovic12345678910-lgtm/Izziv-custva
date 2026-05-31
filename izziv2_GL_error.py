from pathlib import Path
import csv


BASE_DIR = Path(__file__).resolve().parent
INTERIM_DIR = BASE_DIR / "data/interim"
PROCESSED_DIR = BASE_DIR / "data/processed"

GL_INPUT = INTERIM_DIR / "izziv2_GL_all_normalized.csv"
G1_INPUT = INTERIM_DIR / "anketa_G1_vzburjenje_normalized.csv"
OUTPUT = PROCESSED_DIR / "izziv2_GL_emotion_error.csv"
WEIGHTS_OUTPUT = PROCESSED_DIR / "izziv2_GL_emotion_weights.csv"

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


def load_weights():
    """Use absolute valence distance from neutral, then normalize to sum to 1."""
    distances = {}

    with G1_INPUT.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            emotion = row["GL čustvo"].strip()
            if emotion not in EMOTIONS:
                continue
            valence = float(row["Valenca"])
            distances[emotion] = abs(valence - 0.5)

    missing = [emotion for emotion in EMOTIONS if emotion not in distances]
    if missing:
        raise ValueError(f"Missing G1 weights for emotions: {', '.join(missing)}")

    total = sum(distances.values())
    if total == 0:
        raise ValueError("All emotion weights are zero; cannot normalize weights.")

    return {emotion: distance / total for emotion, distance in distances.items()}


GROUND_TRUTH_BY_CLIP = {
    "ples": "happiness",
    "hodnik": "fear",
    "restavracija": "amusement",
}


def baseline_for(clip, emotion):
    target_emotion = GROUND_TRUTH_BY_CLIP[clip]
    return 1.0 if emotion == target_emotion else 0.0


def calculate_error(row, weights):
    error = 0.0
    clip = row["klip"]
    for emotion in EMOTIONS:
        actual = float(row[emotion])
        baseline = baseline_for(clip, emotion)
        error += weights[emotion] * (actual - baseline) ** 2
    return error


def main():
    weights = load_weights()
    output_rows = []

    with GL_INPUT.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            error = calculate_error(row, weights)
            output_rows.append(
                {
                    "klip": row["klip"],
                    "original method": row["original method"],
                    "šifra": row["šifra"],
                    "emotion_error": f"{error:.6f}".rstrip("0").rstrip("."),
                }
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["klip", "original method", "šifra", "emotion_error"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    with WEIGHTS_OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["emotion", "weight"])
        writer.writeheader()
        for emotion in EMOTIONS:
            writer.writerow(
                {
                    "emotion": emotion,
                    "weight": f"{weights[emotion]:.6f}".rstrip("0").rstrip("."),
                }
            )

    print(f"Wrote {len(output_rows)} rows to {OUTPUT}")
    print(f"Wrote {len(weights)} weights to {WEIGHTS_OUTPUT}")


if __name__ == "__main__":
    main()
