import unittest
import torch
import numpy as np

from .mrna import MRNADesignTask, AA_TO_CODONS, CODON_TO_IDX, IDX_TO_CODON

# --- Mock Tokenizer for testing ---
class MockCodonTokenizer:
    def __init__(self):
        self.codon_to_id = CODON_TO_IDX
        self.id_to_codon = IDX_TO_CODON

    def convert_token_to_id(self, token: str) -> int:
        # The MOGFN model will likely treat codons as tokens.
        # It adds special tokens, so let's assume a simple mapping for the test
        # where the model's action `i` corresponds to codon with index `i`.
        # This will be replaced by the actual tokenizer logic in Part 2.
        return self.codon_to_id.get(token, -1)

    def convert_id_to_token(self, id: int) -> str:
        return self.id_to_codon.get(id, "[UNK]")

class TestMRNADesignTask(unittest.TestCase):

    def setUp(self):
        """Set up a task instance for testing."""
        self.protein_seq = "MFV" # Methionine, Phenylalanine, Valine
        self.tokenizer = MockCodonTokenizer()
        self.task = MRNADesignTask(
            protein_seq=self.protein_seq,
            tokenizer=self.tokenizer,
            device="cpu"
        )
        # Manually map codon IDs for the test
        self.aa_to_codon_ids = {
            aa: [self.tokenizer.convert_token_to_id(c) for c in codons]
            for aa, codons in AA_TO_CODONS.items()
        }


    def test_get_valid_actions(self):
        """Test that correct valid codons are returned for each position."""
        # Position 0: Methionine (M)
        valid_m = self.task.get_valid_actions(0)
        expected_m = self.aa_to_codon_ids['M']
        self.assertCountEqual(valid_m, expected_m, "Should return codons for Methionine")

        # Position 1: Phenylalanine (F)
        valid_f = self.task.get_valid_actions(1)
        expected_f = self.aa_to_codon_ids['F']
        self.assertCountEqual(valid_f, expected_f, "Should return codons for Phenylalanine")

        # Position 2: Valine (V)
        valid_v = self.task.get_valid_actions(2)
        expected_v = self.aa_to_codon_ids['V']
        self.assertCountEqual(valid_v, expected_v, "Should return codons for Valine")
        
        # Position 3: Out of bounds
        self.assertEqual(self.task.get_valid_actions(3), [], "Should return empty list for out-of-bounds position")

    def test_score_method(self):
        """Test the score method for correct output shape and value range."""
        # Create a sample sequence of codon names based on the protein
        # Note: the score function expects space-separated string of codons
        seq1_codons = [AA_TO_CODONS['M'][0], AA_TO_CODONS['F'][0], AA_TO_CODONS['V'][0]]
        seq1_str = " ".join(seq1_codons)

        seq2_codons = [AA_TO_CODONS['M'][0], AA_TO_CODONS['F'][1], AA_TO_CODONS['V'][1]]
        seq2_str = " ".join(seq2_codons)
        
        sequences = [seq1_str, seq2_str]
        
        scores = self.task.score(sequences)

        # Check shape
        self.assertEqual(scores.shape, (2, 3), "Scores should have shape (batch_size, num_objectives)")

        # Check value range (all scores should be normalized between 0 and 1)
        self.assertTrue(np.all(scores >= 0) and np.all(scores <= 1), 
                        f"All scores must be in [0, 1], but got: \\n{scores}")

    def test_score_empty_sequence(self):
        """Test that an empty sequence is handled gracefully."""
        scores = self.task.score([""])
        self.assertEqual(scores.shape, (1, 3))
        self.assertTrue(np.all(scores == 0))

if __name__ == "__main__":
    unittest.main() 