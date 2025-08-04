import torch

# --- Constants for mRNA Design ---

CODONS = {
    'AUG': 'M',   
    'UCU': 'S',   
    'UAC': 'Y',   
    'UGA': '*',   
    'CUG': 'L',   
    'CAU': 'H',   
    'CGC': 'R',   
    'AUA': 'I',    
    'AAU': 'N',    
    'GUU': 'V',   
    'GCC': 'A',   
    'GAA': 'E',   
    'GGG': 'G',  
    'UUU': 'F',
    'UCC': 'S',
    'UAA': '*',
    'UGG': 'W',
    'CCU': 'P',
    'CAC': 'H',
    'CGA': 'R',
    'ACU': 'T',
    'AAC': 'N',
    'GUC': 'V',
    'GCA': 'A',
    'GAG': 'E',
    'AGC': 'S',
    'UUC': 'F',
    'UCA': 'S',
    'UAG': '*',
    'CUU': 'L',
    'CCC': 'P',    
    'CAA': 'Q',    
    'CGG': 'R',    
    'ACC': 'T',    
    'AAA': 'K',    
    'GUA': 'V',    
    'GCG': 'A',
    'GGU': 'G',
    'AGA': 'R',
    'UUA': 'L',   
    'UCG': 'S',   
    'UGU': 'C',   
    'CUC': 'L',   
    'CCA': 'P',   
    'CAG': 'Q',   
    'AUU': 'I',   
    'ACA': 'T',   
    'AAG': 'K',   
    'GUG': 'V',   
    'GAU': 'D',   
    'GGC': 'G',   
    'AGG': 'R',   
    'UUG': 'L',   
    'UAU': 'Y',   
    'UGC': 'C',   
    'CUA': 'L',   
    'CCG': 'P',   
    'CGU': 'R',   
    'AUC': 'I',   
    'ACG': 'T',
    'AGU': 'S',       
    'GCU': 'A',   
    'GAC': 'D',   
    'GGA': 'G',
}

STOP_CODONS = {k: v for k, v in CODONS.items() if v == '*'}

SENSE_CODONS = {k: v for k, v in CODONS.items() if v != '*'}

# Inverse mapping from amino acids to codons
AA_TO_CODONS = {}
for codon, aa in SENSE_CODONS.items():
    if aa not in AA_TO_CODONS:
        AA_TO_CODONS[aa] = []
    AA_TO_CODONS[aa].append(codon)

# Codon to integer index mapping
CODON_TO_IDX = {codon: i for i, codon in enumerate(SENSE_CODONS.keys())}
IDX_TO_CODON = {i: codon for codon, i in CODON_TO_IDX.items()}
N_CODONS = len(SENSE_CODONS)

# Precompute GC counts for each codon
CODON_GC_COUNTS = torch.tensor([
    c.count('G') + c.count('C') for c in CODON_TO_IDX.keys()
]) 

N_CODONS = len(CODONS)