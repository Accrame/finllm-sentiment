# FinLLM-Sentiment

Fine-tuning LLMs (Mistral, Llama) for financial sentiment analysis using QLoRA. The goal is to get better sentiment classification than FinBERT while being able to generate explanations — something traditional classifiers can't do.

## Why this project?

After working with classical ML for a while, I wanted to try fine-tuning actual LLMs for a domain-specific task. Financial sentiment seemed like a good fit because:
- FinBERT exists as a strong baseline to compare against
- The domain has nuanced language that base LLMs get wrong ("revenue missed but guidance raised" — positive or negative?)
- QLoRA makes it possible to fine-tune 7B models on a single GPU

I waited for Llama 3 and Mistral to mature before starting, so I could use recent models with good tokenizers.

## How it works

1. Load Financial PhraseBank or FiQA dataset
2. Format examples as instruction prompts (text → sentiment label)
3. Fine-tune with LoRA (only ~4M params out of 7B — 0.06%)
4. Evaluate against base model and FinBERT

## Results

| Model | PhraseBank Acc | FiQA Acc |
|-------|---------------|----------|
| FinBERT (baseline) | 87.2% | 85.1% |
| Mistral-7B (no fine-tuning) | 72.4% | 68.9% |
| **Mistral + LoRA** | **89.8%** | **88.3%** |
| Llama-3-8B (no fine-tuning) | 74.1% | 70.2% |
| **Llama-3 + LoRA** | **90.2%** | **89.1%** |

Training takes ~2 hours on a single A100 with 4-bit quantization (~16GB VRAM).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

huggingface-cli login  # need access to gated models
```

## Training

```bash
# with config file
python scripts/train.py --config configs/lora_mistral.yaml

# or CLI args
python scripts/train.py --base-model mistralai/Mistral-7B-v0.1 --epochs 3 --batch-size 4
```

## Inference

```python
from src.inference.predictor import SentimentPredictor

predictor = SentimentPredictor("./outputs/finllm-sentiment-lora")
result = predictor.predict("Apple beat revenue estimates by 3%")
# {'sentiment': 'positive', 'confidence': 0.92}
```

There's also a Streamlit demo (`streamlit run streamlit_app/app.py`) but it currently uses a keyword-based placeholder — haven't hooked up the actual model yet.

## Lessons learned

- **CUDA OOM is real**: Had to drop batch size from 8 to 4 and crank up gradient accumulation. `paged_adamw_32bit` optimizer helped a lot with memory.
- **Prompt format matters a lot**: Spent weeks trying different templates. Simple classification prompts work best for training, but JSON format is better at inference time.
- **Pad token debugging**: Mistral doesn't have a pad token by default. Setting it to EOS and making sure `pad_token_id` is set everywhere took embarrassingly long to figure out. Loss was going to NaN and I had no idea why.
- **Financial language is tricky**: "Revenue missed expectations but guidance raised" — models disagree on this one. These edge cases are where fine-tuning actually helps vs just prompting.

## What I'd do differently

- Try DPO (Direct Preference Optimization) instead of SFT — might help with the ambiguous cases
- Use a bigger evaluation set. PhraseBank is small and FiQA even smaller
- The inference pipeline is kind of hacky (regex-based output parsing). Would be cleaner with structured generation
- Batch inference is just a loop right now, should properly batch the generation
