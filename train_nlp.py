import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification

# 1. Setup paths and parameters
DATA_PATH = "./processed_data/cleaned_nlp_data.csv"
MODEL_SAVE_DIR = "./saved_models"
BATCH_SIZE = 16
EPOCHS = 1  
LEARNING_RATE = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on: {device}")


# 2. Custom Dataset for BERT
class BurnoutDataset(Dataset):

    def __init__(self, texts, scores, tokenizer, max_len=64):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        score = float(self.scores[item])

        #Called tokenizer directly
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "score": torch.tensor(score, dtype=torch.float),
        }


# 3. Main execution
if not os.path.exists(DATA_PATH):
    print(f"❌ Cleaned data not found at {DATA_PATH}!")
else:
    print("📂 Loading cleaned dataset...")

    # Grabbing the first 15,000 rows as requested!
    df = pd.read_csv(DATA_PATH).head(15000)

    df["cleaned_text"] = df["cleaned_text"].fillna("")

    texts = df["cleaned_text"].values
    scores = df["burnout_score"].values

    print("🧠 Loading BERT Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset = BurnoutDataset(texts, scores, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("🤖 Loading pre-trained BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=1
    )
    model.to(device)

    # Using PyTorch's native AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("\n🏋️ Starting BERT Fine-Tuning...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            scores_tensor = batch["score"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=scores_tensor.unsqueeze(1),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(
                    f"Batch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f}"
                )

    # Save the final model
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, "bert_burnout_model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"💾 Success! BERT model saved in {MODEL_SAVE_DIR}")

    # 4. SIMULATED DASHBOARD OUTPUT
    print("\n" + "=" * 40)
    print("🖥️  SIMULATED APP DASHBOARD PREVIEW")
    print("=" * 40)

    test_comment = "I feel so overwhelmed and exhausted every day."
    print(f"User Inputted Survey Text: '{test_comment}'")

    model.eval()
    inputs = tokenizer(
        test_comment, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        output_score = model(**inputs).logits.item()

    # Bound the score between 0 and 1
    output_score = max(0.0, min(1.0, output_score))

    if output_score <= 0.3:
        status = "🟢 LOW RISK"
    elif output_score <= 0.69:
        status = "🟡 MODERATE RISK"
    else:
        status = "🔴 HIGH RISK"

    print(f"Calculated Stress Score: {output_score*100:.1f}%")
    print(f"Dashboard Status: {status}")
    print("=" * 40)