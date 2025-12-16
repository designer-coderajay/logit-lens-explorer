# Logit Lens Explorer

A mechanistic interpretability tool that reveals what GPT-2 is "thinking" at each layer.

## The Core Idea

At the final layer, GPT-2 converts its internal representation to predictions:

```
logits = LayerNorm(residual_stream) @ W_U
```

The **logit lens** applies this same operation to *intermediate* layers. Why does this make sense?

### Why Residual Streams Make This Work

Transformers use residual connections. Each layer *adds* to the stream rather than replacing it:

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

## Key Concepts for Interview Prep

### Q: What is the logit lens?

**A**: A technique that applies the model's final unembedding operation to intermediate layer representations. This reveals what the model would predict if we stopped computation at that layer. It works because residual connections mean information accumulates gradually, so each layer has a "partial answer" that we can decode.

### Q: Why do we apply LayerNorm?

**A**: GPT-2's architecture applies LayerNorm before the unembedding matrix. The model learned its unembedding weights expecting normalized inputs. Skipping LayerNorm would give us unscaled, meaningless logits. This is architecture-specific. Some models (like LLaMA) use RMSNorm, others have different conventions.

### Q: What does it mean if a prediction appears early?

**A**: The relevant information is easily extractable from the embeddings and early attention patterns. For "The Eiffel Tower is in", the association between "Eiffel Tower" and "Paris" is likely stored in the embedding space or learned in early layers.

### Q: What does it mean if a prediction changes late?

**A**: Later layers are doing something more than simple retrieval. They might be:
- Resolving ambiguity based on context
- Composing information from multiple sources
- Applying learned heuristics or reasoning patterns

### Q: How is this related to "iterative inference"?

**A**: One interpretation of transformers is that each layer performs a step of iterative inference, refining the model's "belief" about what comes next. The logit lens makes this visible. You can literally watch the probability distribution sharpen across layers.

## Further Reading

- [Interpreting GPT: The Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) - Original blog post by nostalgebraist
- [TransformerLens Documentation](https://neelnanda-io.github.io/TransformerLens/)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) - Anthropic's foundational paper

## License

MIT License
