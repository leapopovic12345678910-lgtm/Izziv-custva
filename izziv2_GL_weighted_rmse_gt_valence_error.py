from pathlib import Path
import csv
import math


BASE_DIR = Path(__file__).resolve().parent
INTERIM_DIR = BASE_DIR / "data/interim"
PROCESSED_DIR = BASE_DIR / "data/processed"

GL_INPUT = INTERIM_DIR / "izziv2_GL_all_normalized.csv"
G1_INPUT = INTERIM_DIR / "anketa_G1_vzburjenje_normalized.csv"
OUTPUT = PROCESSED_DIR / "izziv2_GL_weighted_rmse_gt_valence_error.csv"
WEIGHTS_OUTPUT = PROCESSED_DIR / "izziv2_GL_weighted_rmse_gt_valence_weights.csv"
ERROR_METHOD = "weighted_rmse_gt_valence_distance"

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


def format_number(value):
    return f"{value:.6f}".rstrip("0").rstrip(".")


def load_valences():
    valences = {}

    with G1_INPUT.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            row = clean_row(raw_row)
            emotion = row["GL čustvo"]
            if emotion in EMOTIONS:
                valences[emotion] = float(row["Valenca"])

    missing = [emotion for emotion in EMOTIONS if emotion not in valences]
    if missing:
        raise ValueError(f"Missing G1 valence for emotions: {', '.join(missing)}")

    return valences


def load_clip_weights():
    valences = load_valences()
    weights_by_clip = {}

    for clip, target_emotion in GROUND_TRUTH_BY_CLIP.items():
        target_valence = valences[target_emotion]
        raw_weights = {}
        for emotion, valence in valences.items():
            if emotion == target_emotion:
                raw_weights[emotion] = 1.0
            else:
                raw_weights[emotion] = abs(target_valence - valence)

        total = sum(raw_weights.values())
        if total == 0:
            raise ValueError(f"All weights are zero for clip: {clip}")
        weights_by_clip[clip] = {
            emotion: raw_weight / total
            for emotion, raw_weight in raw_weights.items()
        }

    return weights_by_clip


def target_value(clip, emotion):
    return 1.0 if emotion == GROUND_TRUTH_BY_CLIP[clip] else 0.0


def calculate_weighted_rmse(row, weights_by_clip):
    weighted_squared_error = 0.0
    clip = row["klip"]
    weights = weights_by_clip[clip]

    for emotion in EMOTIONS:
        actual = float(row[emotion])
        target = target_value(clip, emotion)
        weighted_squared_error += weights[emotion] * (actual - target) ** 2

    return math.sqrt(weighted_squared_error)


def write_weights(weights_by_clip):
    with WEIGHTS_OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["klip", "target_emotion", "emotion", "weight"],
        )
        writer.writeheader()
        for clip in GROUND_TRUTH_BY_CLIP:
            for emotion in EMOTIONS:
                writer.writerow(
                    {
                        "klip": clip,
                        "target_emotion": GROUND_TRUTH_BY_CLIP[clip],
                        "emotion": emotion,
                        "weight": format_number(weights_by_clip[clip][emotion]),
                    }
                )


def main():
    weights_by_clip = load_clip_weights()
    output_rows = []

    with GL_INPUT.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            row = clean_row(raw_row)
            error = calculate_weighted_rmse(row, weights_by_clip)
            output_rows.append(
                {
                    "klip": row["klip"],
                    "original method": row["original method"],
                    "šifra": row["šifra"],
                    "error_method": ERROR_METHOD,
                    "emotion_error": format_number(error),
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

    write_weights(weights_by_clip)

    print(f"Wrote {len(output_rows)} rows to {OUTPUT}")
    print(f"Wrote {sum(len(weights) for weights in weights_by_clip.values())} weights to {WEIGHTS_OUTPUT}")


if __name__ == "__main__":
    main()
