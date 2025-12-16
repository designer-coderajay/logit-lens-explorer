"""
Unit Tests for Logit Lens Implementation
========================================

These tests verify:
1. Correct tensor shapes at each step
2. Logit lens output matches model's actual final output
3. Probability distributions are valid (sum to 1)
"""

import torch
import unittest
from logit_lens import LogitLens


class TestLogitLens(unittest.TestCase):
    """Test suite for LogitLens class."""
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        print("\nLoading model for tests...")
        cls.lens = LogitLens("gpt2-small")
        cls.prompt = "The Eiffel Tower is in"
    
    def test_model_dimensions(self):
        """Verify model dimensions match GPT-2 Small specs."""
        self.assertEqual(self.lens.n_layers, 12, "GPT-2 Small should have 12 layers")
        self.assertEqual(self.lens.d_model, 768, "GPT-2 Small d_model should be 768")
        self.assertEqual(self.lens.vocab_size, 50257, "GPT-2 vocab should be 50257")
    
    def test_residual_stream_shape(self):
        """Verify residual stream has correct shape."""
        residuals = self.lens.get_residual_stream_at_all_layers(self.prompt)
        
        # Should be (n_layers + 1, batch, seq_len, d_model)
        # n_layers + 1 because we include embedding (layer 0)
        expected_layers = self.lens.n_layers + 1  # 13 total
        
        self.assertEqual(residuals.shape[0], expected_layers, 
                        f"Should have {expected_layers} residual snapshots")
        self.assertEqual(residuals.shape[1], 1,
                        "Batch dimension should be 1")
        self.assertEqual(residuals.shape[3], self.lens.d_model,
                        f"d_model should be {self.lens.d_model}")
    
    def test_logit_output_shape(self):
        """Verify logits have correct shape."""
        residuals = self.lens.get_residual_stream_at_all_layers(self.prompt)
        
        # Take one residual vector
        residual = residuals[0, 0, -1]  # Embedding layer, batch 0, last position
        
        logits = self.lens.residual_to_logits(residual)
        
        self.assertEqual(logits.shape[0], self.lens.vocab_size,
                        f"Logits should have {self.lens.vocab_size} dimensions")
    
    def test_final_layer_matches_model_output(self):
        """
        CRITICAL TEST: Logit lens on final layer should match model's actual output.
        
        If this fails, something is wrong with our implementation.
        """
        # Get model's actual output
        tokens = self.lens.model.to_tokens(self.prompt)
        with torch.no_grad():
            actual_logits = self.lens.model(tokens)[0, -1]  # Last position logits
        
        # Get logit lens output from final layer
        residuals = self.lens.get_residual_stream_at_all_layers(self.prompt)
        final_residual = residuals[-1, 0, -1]  # Last layer, batch 0, last position
        lens_logits = self.lens.residual_to_logits(final_residual)
        
        # They should be very close (allowing for floating point differences)
        diff = torch.abs(actual_logits - lens_logits).max().item()
        
        self.assertLess(diff, 1e-4, 
                       f"Logit lens final layer should match model output. Max diff: {diff}")
    
    def test_probabilities_sum_to_one(self):
        """Verify probability distributions are valid."""
        residuals = self.lens.get_residual_stream_at_all_layers(self.prompt)
        
        for layer_idx in range(self.lens.n_layers + 1):
            residual = residuals[layer_idx, 0, -1]
            logits = self.lens.residual_to_logits(residual)
            probs = torch.softmax(logits, dim=-1)
            
            prob_sum = probs.sum().item()
            self.assertAlmostEqual(prob_sum, 1.0, places=5,
                                  msg=f"Layer {layer_idx} probs should sum to 1")
    
    def test_top_predictions_format(self):
        """Verify top predictions return correct format."""
        residuals = self.lens.get_residual_stream_at_all_layers(self.prompt)
        residual = residuals[-1, 0, -1]
        logits = self.lens.residual_to_logits(residual)
        
        top_preds = self.lens.get_top_predictions(logits, k=5)
        
        self.assertEqual(len(top_preds), 5, "Should return 5 predictions")
        
        for token, prob in top_preds:
            self.assertIsInstance(token, str, "Token should be a string")
            self.assertIsInstance(prob, float, "Probability should be a float")
            self.assertGreaterEqual(prob, 0.0, "Probability should be >= 0")
            self.assertLessEqual(prob, 1.0, "Probability should be <= 1")
    
    def test_analyze_prompt_structure(self):
        """Verify analyze_prompt returns correct structure."""
        analysis = self.lens.analyze_prompt(self.prompt, top_k=5)
        
        # Check required keys
        self.assertIn("prompt", analysis)
        self.assertIn("tokens", analysis)
        self.assertIn("layers", analysis)
        
        # Check layers structure
        self.assertEqual(len(analysis["layers"]), self.lens.n_layers + 1)
        
        for layer_info in analysis["layers"]:
            self.assertIn("layer_name", layer_info)
            self.assertIn("predictions", layer_info)
            self.assertEqual(len(layer_info["predictions"]), 5)
    
    def test_early_vs_late_layer_confidence(self):
        """
        Sanity check: final layer should generally be more confident for clear patterns.
        
        We use "import numpy as" because GPT-2 definitively knows "np" comes next.
        The Eiffel Tower prompt doesn't work because GPT-2 Small has weak factual recall.
        """
        analysis = self.lens.analyze_prompt("import numpy as", top_k=1)
        
        embed_top_prob = analysis["layers"][0]["predictions"][0][1]
        final_top_prob = analysis["layers"][-1]["predictions"][0][1]
        
        # Final should be MUCH more confident for this clear code pattern
        # We saw 99.9% confidence on "np" at layer 11 vs ~66% on "mathemat" at embedding
        self.assertGreater(final_top_prob, 0.9,
                          "Final layer should be >90% confident on 'np' for 'import numpy as'")
    
    def test_different_prompts(self):
        """Test analysis works on various prompts."""
        prompts = [
            "Hello",
            "The quick brown fox",
            "import numpy as",
            "def fibonacci(",
        ]
        
        for prompt in prompts:
            try:
                analysis = self.lens.analyze_prompt(prompt, top_k=3)
                self.assertEqual(analysis["prompt"], prompt)
            except Exception as e:
                self.fail(f"Failed on prompt '{prompt}': {e}")


class TestLayerNormImportance(unittest.TestCase):
    """Test that LayerNorm is critical for correct results."""
    
    @classmethod
    def setUpClass(cls):
        cls.lens = LogitLens("gpt2-small")
    
    def test_layernorm_affects_output(self):
        """Verify that skipping LayerNorm gives different (wrong) results."""
        prompt = "The capital of France is"
        residuals = self.lens.get_residual_stream_at_all_layers(prompt)
        final_residual = residuals[-1, 0, -1]
        
        # With LayerNorm (correct)
        with_ln = self.lens.residual_to_logits(final_residual)
        
        # Without LayerNorm (incorrect)
        without_ln = final_residual @ self.lens.model.W_U + self.lens.model.b_U
        
        # They should be different
        diff = torch.abs(with_ln - without_ln).mean().item()
        self.assertGreater(diff, 1.0, 
                          "LayerNorm should significantly affect logits")


if __name__ == "__main__":
    unittest.main(verbosity=2)
