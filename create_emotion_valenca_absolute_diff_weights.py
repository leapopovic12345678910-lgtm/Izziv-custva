from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/interim/anketa_G1_vzburjenje_normalized.csv")
OUTPUT_PATH = Path("data/interim/emotion_arousal_absolute_diff_weights_fixed.csv")

TARGET_EMOTIONS = ["happiness", "fear", "amusement"]

def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    emotion_col = "GL čustvo"
    arousal_col = "Valenca"

    required_cols = {emotion_col, arousal_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df[emotion_col] = df[emotion_col].astype(str).str.strip().str.lower()

    arousal_by_emotion = dict(zip(df[emotion_col], df[arousal_col]))

    missing_targets = [emotion for emotion in TARGET_EMOTIONS if emotion not in arousal_by_emotion]
    if missing_targets:
        raise ValueError(f"Target emotions not found in input data: {missing_targets}")

    rows = []
    for target_emotion in TARGET_EMOTIONS:
        target_arousal = arousal_by_emotion[target_emotion]

        for _, row in df.iterrows():
            emotion = row[emotion_col]
            emotion_arousal = row[arousal_col]

            rows.append({
                "target_emotion": target_emotion,
                "emotion": emotion,
                "weight": abs(target_arousal - emotion_arousal),
            })

    result = pd.DataFrame(rows, columns=["target_emotion", "emotion", "weight"])
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Created: {OUTPUT_PATH}")
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
