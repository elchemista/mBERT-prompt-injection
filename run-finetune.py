# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "torch==2.5.1",
#     "transformers>=4.46,<5",
# ]
# ///

import sys, json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path(__file__).resolve().parent / "mbert-pi"  # hardcoded


def load_model():
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32
    )
    model.to("cpu").eval()
    id2label = getattr(model.config, "id2label", {0: "0", 1: "1"})
    # normalize keys to int
    id2label = {
        int(k) if isinstance(k, str) and k.isdigit() else k: v
        for k, v in id2label.items()
    }
    return tok, model, id2label


def predict(texts, tok, model, max_length=512):
    enc = tok(
        texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
    return preds, probs


def main():
    # Input texts: args or stdin lines
    texts = sys.argv[1:] or [ln.strip() for ln in sys.stdin if ln.strip()]
    if not texts:
        print("Provide text as arguments or via stdin.", file=sys.stderr)
        sys.exit(1)

    tok, model, id2label = load_model()
    preds, probs = predict(texts, tok, model, max_length=512)

    for t, p, pr in zip(texts, preds, probs):
        p = int(p)
        label = id2label.get(p, id2label.get(str(p), str(p)))
        print(
            json.dumps(
                {"label_id": p, "label": label, "score": float(pr[p]), "text": t},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
