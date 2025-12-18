# Logit Lens Explorer

A mechanistic interpretability tool that reveals what GPT-2 is "thinking" at each layer.

## The Core Idea

At the final layer, GPT-2 converts its internal representation to predictions:

```
logits = LayerNorm(residual_stream) @ W_U
```

The **logit lens** applies this same operation to _intermediate_ layers. Why does this make sense?

### Why Residual Streams Make This Work

Transformers use residual connections. Each layer _adds_ to the stream rather than replacing it:

```
residual_after = residual_before + layer_output
```

This means information accumulates gradually. The model builds its prediction across layers. By projecting intermediate states into vocabulary space, we watch this buildup happen.

## What You'll See

Run the demo:

```bash
python logit_lens.py
```

Output for "The Eiffel Tower is in":

```
Embedding    |  the: 0.045 |  a: 0.032 | ...
Layer 0      |  the: 0.052 |  France: 0.028 | ...
Layer 1      |  France: 0.089 |  Paris: 0.067 | ...
...
Layer 11     |  Paris: 0.412 |  France: 0.198 | ...
```

### Patterns to Look For

1. **Gradual Refinement**: Model starts uncertain, becomes confident
2. **Early Knowing**: Answer appears early (easy pattern)
3. **Late Corrections**: Model changes prediction in late layers (complex reasoning)
4. **Confidence Jumps**: Specific layers where key computations happen

## Interactive Explorer

```bash
streamlit run app.py
```

Features:

- Type any prompt and see layer-by-layer predictions
- Heatmap of confidence across layers
- Track specific token probability evolution
- One-click example prompts

## Technical Details

### The Three Key Components

1. **Residual Stream** (`hook_resid_post`)

   - The main highway of information flow
   - Shape: `(batch, seq_len, d_model)` = `(1, n_tokens, 768)` for GPT-2 Small

2. **Unembedding Matrix** (`model.W_U`)

   - Converts hidden states to vocabulary logits
   - Shape: `(d_model, vocab_size)` = `(768, 50257)` for GPT-2 Small

3. **Final LayerNorm** (`model.ln_final`)
   - GPT-2 applies LayerNorm before unembedding
   - Critical for proper logit scaling

### The Projection Operation

```python
def residual_to_logits(residual):
    normalized = model.ln_final(residual)  # Don't skip this!
    logits = normalized @ model.W_U + model.b_U
    return logits
```

### Why LayerNorm Matters

Without LayerNorm, the magnitude of residual streams varies wildly across layers. Early layers have smaller magnitudes, late layers have larger ones. LayerNorm standardizes this, making cross-layer comparisons meaningful.

## Files

```
logit-lens-explorer/
├── logit_lens.py      # Core implementation with detailed comments
├── app.py             # Streamlit interactive app
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run demo
python logit_lens.py

# Run interactive app
streamlit run app.py
```

## Further Reading

- [Interpreting GPT: The Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) - Original blog post by nostalgebraist
- [TransformerLens Documentation](https://neelnanda-io.github.io/TransformerLens/)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) - Anthropic's foundational paper

## License

MIT License
