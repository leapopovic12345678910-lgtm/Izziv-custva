from pathlib import Path
import csv
import math


BASE_DIR = Path(__file__).resolve().parent
INTERIM_DIR = BASE_DIR / "data/interim"
PROCESSED_DIR = BASE_DIR / "data/processed"

GL_INPUT = INTERIM_DIR / "izziv2_GL_all_normalized.csv"
WEIGHTS_INPUT = INTERIM_DIR / "emotion_arousal_absolute_diff_weights.csv"
OUTPUT = PROCESSED_DIR / "izziv2_GL_weighted_rmse_arousal_diff_error.csv"
WEIGHTS_OUTPUT = PROCESSED_DIR / "izziv2_GL_weighted_rmse_arousal_diff_weights.csv"
ERROR_METHOD = "weighted_rmse_arousal_absolute_diff"

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


def load_weights():
    raw_weights = {}

    with WEIGHTS_INPUT.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            row = clean_row(raw_row)
            target_emotion = row["target_emotion"]
            emotion = row["emotion"]
            if target_emotion not in GROUND_TRUTH_BY_CLIP.values():
                continue
            if emotion not in EMOTIONS:
                continue
            raw_weights.setdefault(target_emotion, {})[emotion] = float(row["weight"])

    for target_emotion in GROUND_TRUTH_BY_CLIP.values():
        missing = [
            emotion
            for emotion in EMOTIONS
            if emotion not in raw_weights.get(target_emotion, {})
        ]
        if missing:
            raise ValueError(
                f"Missing weights for target {target_emotion}: {', '.join(missing)}"
            )

    normalized_weights = {}
    for target_emotion, weights in raw_weights.items():
        total = sum(weights.values())
        if total == 0:
            raise ValueError(f"All weights are zero for target: {target_emotion}")
        normalized_weights[target_emotion] = {
            emotion: weight / total
            for emotion, weight in weights.items()
        }

    return normalized_weights


def target_value(clip, emotion):
    return 1.0 if emotion == GROUND_TRUTH_BY_CLIP[clip] else 0.0


def calculate_weighted_rmse(row, weights_by_target):
    weighted_squared_error = 0.0
    clip = row["klip"]
    target_emotion = GROUND_TRUTH_BY_CLIP[clip]
    weights = weights_by_target[target_emotion]

    for emotion in EMOTIONS:
        actual = float(row[emotion])
        target = target_value(clip, emotion)
        weighted_squared_error += weights[emotion] * (actual - target) ** 2

    return math.sqrt(weighted_squared_error)


def write_weights(weights_by_target):
    with WEIGHTS_OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["target_emotion", "emotion", "weight"],
        )
        writer.writeheader()
        for target_emotion in ["happiness", "fear", "amusement"]:
            for emotion in EMOTIONS:
                writer.writerow(
                    {
                        "target_emotion": target_emotion,
                        "emotion": emotion,
                        "weight": format_number(weights_by_target[target_emotion][emotion]),
                    }
                )


def main():
    weights_by_target = load_weights()
    output_rows = []

    with GL_INPUT.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            row = clean_row(raw_row)
            error = calculate_weighted_rmse(row, weights_by_target)
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

    write_weights(weights_by_target)

    print(f"Wrote {len(output_rows)} rows to {OUTPUT}")
    print(f"Wrote {len(GROUND_TRUTH_BY_CLIP) * len(EMOTIONS)} weights to {WEIGHTS_OUTPUT}")


if __name__ == "__main__":
    main()
