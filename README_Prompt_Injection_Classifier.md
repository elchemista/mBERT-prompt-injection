# Prompt-Injection Classifier with ModernBERT

## What it does
Fine-tunes `answerdotai/ModernBERT-base` to detect prompt injection.
Outputs a single label:

- `0` = not_injection
- `1` = injection

## Data
- Primary: `xTRam1/safe-guard-prompt-injection` (has train/test). Expected columns: `text`, `label`.
- Optional negatives: `allenai/wildguardmix` (use non-adversarial prompts as label 0). Disabled in the notebook for simplicity.

## How it works
1. Install `transformers`, `datasets`, `accelerate`.
2. Load dataset and keep only `text` and `label`.
3. Tokenize with ModernBERT tokenizer.
4. Train a `AutoModelForSequenceClassification` head with 2 labels.
5. Evaluate with accuracy and F1.
6. Run a quick inference.

## Run
Open the notebook and execute cells top-to-bottom. GPU is recommended but not required for tiny tests.

## Adjustments
- Increase `num_train_epochs` for better accuracy.
- Turn on the optional cell to add negatives from WildGuardMix.
- Change batch sizes or learning rate if you hit OOM or underfit.

## Notes
- The datasets may contain harmful content. Use with care.
- If your Transformers version lacks ModernBERT, upgrade as indicated in the first cell.
