import torch
import numpy as np
import math
from typing import List
from .. import mfe_calculator
from .. import cai_calculator

from .base import BaseTask
from ..data.mrna_constants import (
    AA_TO_CODONS,
    CODON_GC_COUNTS,
    IDX_TO_CODON,
    CODON_TO_IDX,
)

# --- Objective Functions ---
def to_mRNA_string(indices: torch.LongTensor) -> str:
    """Converts a tensor of codon indices to an mRNA string."""
    return "".join([IDX_TO_CODON[idx.item()] for idx in indices])

def compute_gc_content_vectorized(indices: torch.LongTensor, codon_gc_counts: torch.Tensor) -> torch.FloatTensor:
    """Vectorized GC content calculation using precomputed codon GC counts."""
    if indices.dim() == 0: # handle single index
        indices = indices.unsqueeze(0)
    gc_counts = codon_gc_counts[indices].sum()
    total_nucleotides = indices.shape[0] * 3
    return (gc_counts / total_nucleotides) * 100 if total_nucleotides > 0 else 0.0

def compute_mfe_energy(indices: torch.LongTensor, energies=None, loop_min=4) -> torch.FloatTensor:
    """Compute the minimum free energy (MFE) of an RNA sequence."""
    rna_str = to_mRNA_string(indices)
    try:
        sol = mfe_calculator.RNAFolder(energies=energies, loop_min=loop_min)
        s = sol.solve(rna_str)
        energy = s.energy()
    except Exception as e:
        print(f"Energy computation failed for: {rna_str}, error: {e}")
        energy = float('inf')
    return torch.tensor([energy], dtype=torch.float32)

def compute_cai(indices: torch.LongTensor) -> torch.FloatTensor:
    """Compute the Codon Adaptation Index (CAI)."""
    rna_str = to_mRNA_string(indices)
    try:
        # NOTE: CAICalculator might need codon usage weights for a reference organism.
        # This is a simplified placeholder.
        calc = cai_calculator.CAICalculator(rna_str)
        score = calc.compute_cai()
    except Exception as e:
        print(f"CAI computation failed for: {rna_str}, error: {e}")
        score = 0.0 # Return a low score on failure
    return torch.tensor([score], dtype=torch.float32)

class MRNADesignTask(BaseTask):
    def __init__(self, 
                 protein_seq: str, 
                 tokenizer: 'CodonTokenizer',
                 gc_target: float = 57.5, 
                 gc_width: float = 2.5,
                 mfe_min: float = -500.0, 
                 mfe_max: float = 0.0,
                 cai_min: float = 0.1, 
                 cai_max: float = 1.0,
                 max_len: int = None,
                 min_len: int = None,
                 device: str = "cpu"):

        self.max_len = max_len if max_len is not None else len(protein_seq)
        self.min_len = min_len if min_len is not None else len(protein_seq)
        super().__init__(tokenizer=tokenizer, obj_dim=3, max_len=self.max_len)
        self.protein_seq = protein_seq
        self.obj_dim = 3 # Overriding BaseTask's obj_dim just in case
        self.device = device

        # Normalization parameters for objectives
        self.gc_target = gc_target
        self.gc_width = gc_width
        self.mfe_min = mfe_min
        self.mfe_max = mfe_max
        self.cai_min = cai_min
        self.cai_max = cai_max
        
        self.codon_gc_counts = CODON_GC_COUNTS.to(self.device)

        self._build_valid_actions_cache()
        print(self.valid_actions_cache)

    def _build_valid_actions_cache(self):
        """Pre-computes and caches the valid codons for each position."""
        self.valid_actions_cache = []
        print("protein sequence", self.protein_seq)
        for aa in self.protein_seq:
            print("Letter in Protein Sequence:", aa, "Codons:", AA_TO_CODONS[aa])
            codons = AA_TO_CODONS[aa]
            action_indices = [self.tokenizer.convert_token_to_id(c) for c in codons]
            self.valid_actions_cache.append(action_indices)

    def get_valid_actions(self, position: int) -> List[int]:
        """Returns a list of valid action indices for a given sequence position."""
        # print("position", position)
        # print("len of valid_actions_cache", len(self.valid_actions_cache))
        if position < len(self.valid_actions_cache):
            # print("valid_actions_cache[position]", self.valid_actions_cache[position])
            return self.valid_actions_cache[position]
        return []

    def score(self, sequences: List[str]) -> np.ndarray:
        """
        Calculates and normalizes the scores for a batch of mRNA sequences.
        A sequence is a string of codons separated by spaces, e.g., "AUG UGU ...".
        """
        # print("sequences are: ", sequences)
        # print("len of the sequences are: ", len(sequences))
        batch_scores = []
        for seq_str in sequences:
            codons = seq_str.split()
            # print("codons are: ", codons)
            if not codons:
                batch_scores.append(np.zeros(self.obj_dim))
                continue

            indices = torch.tensor([CODON_TO_IDX[c] for c in codons], dtype=torch.long, device=self.device)
            # print("indices are: ", indices)

            # 1. GC Content (target-based reward)
            gc_val = compute_gc_content_vectorized(indices, self.codon_gc_counts).item()
            # print("gc_val is: ", gc_val)

            gc_reward = math.exp(-0.5 * ((gc_val - self.gc_target) / self.gc_width) ** 2)

            # 2. MFE (lower is better, so we invert)
            mfe_val = compute_mfe_energy(indices).item()
            # print("mfe_val is: ", mfe_val)
            mfe_reward = 1.0 - np.clip((mfe_val - self.mfe_min) / (self.mfe_max - self.mfe_min), 0, 1)
            # print("mfe_reward is: ", mfe_reward)


            # 3. CAI (higher is better)
            cai_val = compute_cai(indices).item()
            # print("cai_val is: ", cai_val)
            cai_reward = np.clip((cai_val - self.cai_min) / (self.cai_max - self.cai_min), 0, 1)
            # print("cai_reward is: ", cai_reward)    

            batch_scores.append(np.array([gc_reward, mfe_reward, cai_reward]))

        return np.array(batch_scores) 