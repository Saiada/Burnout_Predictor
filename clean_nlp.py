import os
import re
import pandas as pd

# Define paths
RAW_DATA_PATH = "./raw_data/go_emotions_dataset.csv"
PROCESSED_DIR = "./processed_data"


def clean_text(text):
    """Cleans raw text for NLP processing."""
    if not isinstance(text, str):
        return ""
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove URLs/Links
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # 3. Remove Reddit usernames (e.g., u/username)
    text = re.sub(r"u/\w+", "", text)
    # 4. Remove special characters and numbers, keeping only letters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # 5. Remove extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


if not os.path.exists(RAW_DATA_PATH):
    print(
        f"❌ File not found at {RAW_DATA_PATH}. Please make sure your GoEmotions file is named correctly!"
    )
else:
    print("📂 Loading GoEmotions dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("🧹 Cleaning text...")
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Map emotions to Burnout Risk
    print("🧠 Mapping emotions to burnout risk scores...")

    # High Risk (Score = 1.0)
    high_risk = [
        "anger",
        "annoyance",
        "disappointment",
        "disapproval",
        "fear",
        "grief",
        "nervousness",
        "sadness",
    ]
    # Medium Risk (Score = 0.5)
    med_risk = ["confusion", "embarrassment", "remorse"]

    # Calculate score based on GoEmotion columns (1 means the emotion is present)
    df["burnout_score"] = 0.0

    for emotion in high_risk:
        if emotion in df.columns:
            df["burnout_score"] += df[emotion] * 1.0

    for emotion in med_risk:
        if emotion in df.columns:
            df["burnout_score"] += df[emotion] * 0.5

    # Cap the score at a maximum of 1.0
    df["burnout_score"] = df["burnout_score"].clip(upper=1.0)

    # Create processed directory if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save it
    output_path = os.path.join(PROCESSED_DIR, "cleaned_nlp_data.csv")
    df.to_csv(output_path, index=False)
    print(f"💾 Success! Cleaned data saved to {output_path}")

    # Preview score distribution
    print("\n📊 Burnout Score Preview:")
    print(df["burnout_score"].value_counts(normalize=True))