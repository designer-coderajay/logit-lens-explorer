"""
Logit Lens Implementation for GPT-2 Small
==========================================

WHAT IS LOGIT LENS?
-------------------
The logit lens is a mechanistic interpretability technique that lets us peek
into what a transformer is "thinking" at each layer.

THE KEY INSIGHT:
At the final layer, GPT-2 converts the residual stream into token predictions
by multiplying by the unembedding matrix W_U:

    logits = residual_stream @ W_U

The logit lens applies this SAME operation to intermediate layers. Why does
this make sense? Because of how residual streams work.

WHY RESIDUAL STREAMS MATTER:
----------------------------
In a transformer, each layer ADDS to the residual stream rather than replacing it:

    residual_after_layer_n = residual_before_layer_n + layer_n_output

This means information accumulates. The model builds up its prediction gradually.
By projecting intermediate residual streams into vocabulary space, we watch
this buildup happen in real-time.

WHAT YOU'LL TYPICALLY SEE:
- Early layers: vague, often wrong predictions
- Middle layers: narrowing down to related tokens
- Late layers: confident, correct predictions

Sometimes the model "knows" early. Sometimes it changes its mind. This technique
reveals those dynamics.

TECHNICAL DETAILS:
------------------
- We use TransformerLens's caching to grab residual streams
- The residual stream at position -1 (last token) is what matters for next-token prediction
- We apply LayerNorm before unembedding (GPT-2's architecture requires this)
- The unembedding matrix W_U has shape (d_model, vocab_size)
"""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class LogitLens:
    """
    Applies the logit lens technique to analyze layer-by-layer predictions.
    
    The logit lens works by taking the residual stream at each layer and
    projecting it into vocabulary space using the model's unembedding matrix.
    
    Key components accessed:
    - model.W_U: Unembedding matrix (d_model -> vocab_size)
    - model.ln_final: Final layer norm (applied before unembedding)
    - Residual stream: The running sum that flows through the model
    """
    
    def __init__(self, model_name: str = "gpt2-small"):
        """
        Initialize with a TransformerLens model.
        
        Args:
            model_name: HuggingFace model name. "gpt2-small" = GPT-2 124M params
        """
        print(f"Loading {model_name}...")
        self.model = HookedTransformer.from_pretrained(model_name)
        self.model.eval()  # Inference mode, no dropout
        
        # Store key dimensions for reference
        self.n_layers = self.model.cfg.n_layers  # 12 for GPT-2 Small
        self.d_model = self.model.cfg.d_model    # 768 for GPT-2 Small
        self.vocab_size = self.model.cfg.d_vocab # 50257 for GPT-2
        
        print(f"Loaded: {self.n_layers} layers, d_model={self.d_model}, vocab={self.vocab_size}")
    
    def get_residual_stream_at_all_layers(self, prompt: str) -> torch.Tensor:
        """
        Run the model and capture the residual stream at every layer.
        
        The residual stream is the "main highway" of information flow.
        After layer i, it contains:
            embed(tokens) + attn_0 + mlp_0 + attn_1 + mlp_1 + ... + attn_i + mlp_i
        
        Returns:
            Tensor of shape (n_layers + 1, seq_len, d_model)
            Index 0 = after embedding (before any layers)
            Index i = after layer i-1 (1-indexed layers become 0-indexed)
        """
        # run_with_cache stores all intermediate activations
        # We ask for "resid_post" which is the residual stream AFTER each layer
        _, cache = self.model.run_with_cache(prompt)
        
        # Collect residual streams
        # Layer 0's input is just the embedding
        residual_streams = []
        
        # First: the embedding (before any transformer layers touch it)
        # This is what the model starts with
        residual_streams.append(cache["hook_embed"] + cache["hook_pos_embed"])
        
        # Then: residual stream after each layer
        for layer in range(self.n_layers):
            # "resid_post" = residual stream AFTER this layer's attention and MLP
            residual_streams.append(cache[f"blocks.{layer}.hook_resid_post"])
        
        # Stack into single tensor: (n_layers + 1, seq_len, d_model)
        return torch.stack(residual_streams)
    
    def residual_to_logits(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Convert a residual stream vector to logits over vocabulary.
        
        This is the core of the logit lens. We're doing what the model does
        at its final layer, but applying it to intermediate layers.
        
        The operation:
            logits = LayerNorm(residual) @ W_U + b_U
        
        Args:
            residual: Tensor of shape (..., d_model)
        
        Returns:
            logits: Tensor of shape (..., vocab_size)
        """
        # IMPORTANT: GPT-2 applies LayerNorm before the unembedding
        # If we skip this, the logits won't be properly scaled
        normalized = self.model.ln_final(residual)
        
        # W_U has shape (d_model, vocab_size)
        # We matrix multiply to get logits for each vocabulary token
        logits = normalized @ self.model.W_U + self.model.b_U
        
        return logits
    
    def get_top_predictions(
        self, 
        logits: torch.Tensor, 
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get the top-k predicted tokens and their probabilities.
        
        Args:
            logits: Tensor of shape (vocab_size,)
            k: Number of top predictions to return
        
        Returns:
            List of (token_string, probability) tuples
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get top k
        top_probs, top_indices = torch.topk(probs, k)
        
        # Decode tokens
        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token = self.model.tokenizer.decode([idx])
            results.append((token, prob))
        
        return results
    
    def analyze_prompt(
        self, 
        prompt: str, 
        top_k: int = 5,
        position: int = -1
    ) -> Dict:
        """
        Full logit lens analysis of a prompt.
        
        Args:
            prompt: The text to analyze
            top_k: Number of top predictions per layer
            position: Which token position to analyze (-1 = last token)
        
        Returns:
            Dictionary with:
                - tokens: List of tokens in the prompt
                - layers: List of layer analyses, each containing top predictions
        """
        # Tokenize to show what the model sees
        tokens = self.model.to_tokens(prompt)
        token_strings = [self.model.tokenizer.decode([t]) for t in tokens[0].tolist()]
        
        # Get residual streams at all layers
        residual_streams = self.get_residual_stream_at_all_layers(prompt)
        
        # Analyze each layer
        layer_analyses = []
        
        for layer_idx in range(self.n_layers + 1):
            # Get residual at this layer for the target position
            # Shape: (d_model,)
            residual = residual_streams[layer_idx, 0, position]  # batch=0, position
            
            # Convert to logits
            logits = self.residual_to_logits(residual)
            
            # Get top predictions
            top_preds = self.get_top_predictions(logits, k=top_k)
            
            # Label the layer
            if layer_idx == 0:
                layer_name = "Embedding"
            else:
                layer_name = f"Layer {layer_idx - 1}"  # 0-indexed layers
            
            layer_analyses.append({
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "predictions": top_preds
            })
        
        return {
            "prompt": prompt,
            "tokens": token_strings,
            "analyzed_position": position,
            "layers": layer_analyses
        }
    
    def print_analysis(self, analysis: Dict) -> None:
        """Pretty print a logit lens analysis."""
        print(f"\n{'='*60}")
        print(f"Prompt: {analysis['prompt']}")
        print(f"Tokens: {analysis['tokens']}")
        print(f"Analyzing position: {analysis['analyzed_position']} ({analysis['tokens'][analysis['analyzed_position']]})")
        print(f"{'='*60}\n")
        
        for layer_info in analysis["layers"]:
            print(f"{layer_info['layer_name']:12s} | ", end="")
            
            preds_str = []
            for token, prob in layer_info["predictions"]:
                # Clean up token display
                token_display = repr(token)[1:-1]  # Remove quotes, keep escapes
                preds_str.append(f"{token_display}: {prob:.3f}")
            
            print(" | ".join(preds_str))
        
        print()


def main():
    """Demo the logit lens on a classic example."""
    lens = LogitLens("gpt2-small")
    
    # Classic example: "The Eiffel Tower is in"
    # The model should predict "Paris" or "France"
    prompt = "The Eiffel Tower is in"
    
    print("\n" + "="*60)
    print("LOGIT LENS DEMO")
    print("="*60)
    print("\nThe logit lens shows what the model 'thinks' at each layer.")
    print("Watch how the prediction evolves from vague to confident.\n")
    
    analysis = lens.analyze_prompt(prompt, top_k=5)
    lens.print_analysis(analysis)
    
    # Additional examples to try
    print("\n" + "="*60)
    print("TRY THESE OTHER PROMPTS:")
    print("="*60)
    
    other_prompts = [
        "Barack Obama was the",
        "The capital of France is",
        "import numpy as",
    ]
    
    for p in other_prompts:
        print(f"\n--- {p} ---")
        analysis = lens.analyze_prompt(p, top_k=3)
        for layer_info in analysis["layers"]:
            top_token, top_prob = layer_info["predictions"][0]
            print(f"{layer_info['layer_name']:12s}: {repr(top_token):10s} ({top_prob:.3f})")


if __name__ == "__main__":
    main()
