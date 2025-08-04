import torch
from typing import List, Dict

# from ..data.mrna_constants import SENSE_CODONS
from mo_gfn.torch_seq_moo.data.mrna_constants import SENSE_CODONS

class CodonTokenizer:
    def __init__(self, **kwargs):
        self.pad_token = '[PAD]' # Padding token
        self.eos_token = '[SEP]' # End of Sequence / Stop token
        self.unk_token = '[UNK]' # Unknown token
        
        self.codons = sorted(list(SENSE_CODONS.keys()))
        # print("codons are: ", self.codons)
        self.non_special_vocab = self.codons
        
        self.special_tokens = [self.pad_token, self.eos_token, self.unk_token]
        self.full_vocab = self.special_tokens + self.codons

        self.token_to_id: Dict[str, int] = {token: i for i, token in enumerate(self.full_vocab)}
        self.id_to_token: Dict[int, str] = {i: token for token, i in self.token_to_id.items()}
        # print("token_to_id is: ", self.token_to_id)
        # print()
        # print()
        # print("id_to_token is: ", self.id_to_token)
        
        self.vocab_size = len(self.full_vocab)
        self.padding_idx = self.token_to_id[self.pad_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]

    def get_vocab(self) -> Dict[str, int]:
        return self.token_to_id

    def convert_token_to_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_token_id)

    def convert_id_to_token(self, id: int) -> str:
        return self.id_to_token.get(id, self.unk_token)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.convert_id_to_token(id) for id in ids]

    def encode(self, seq, use_sep=True):
        print(seq)
        seq = ["[CLS]"] + list(seq[:-1])
        seq += ["[SEP]"] if use_sep else []
        return [self.convert_token_to_id(c) for c in seq]


    # def encode(self, seq, use_sep=True, **kwargs):
    #     return self.batch_encode_plus(seq, **kwargs)
    
    def batch_encode_plus(self, batch_text: List[str], padding=True, max_length=None, return_tensors="pt"):
        """
        Tokenizes a batch of sequences. Expected input is a list of space-separated codon strings.
        """
        all_input_ids = []
        for text in batch_text:
            tokens = text.split() if text else []
            input_ids = self.convert_tokens_to_ids(tokens)
            all_input_ids.append(input_ids)
            
        if padding:
            if max_length is None:
                max_length = max(len(ids) for ids in all_input_ids)
            
            padded_input_ids = []
            attention_masks = []
            for ids in all_input_ids:
                padding_length = max_length - len(ids)
                padded_ids = ids + [self.pad_token_id] * padding_length
                attention_mask = [1] * len(ids) + [0] * padding_length
                padded_input_ids.append(padded_ids)
                attention_masks.append(attention_mask)
            
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
                }
            else:
                return {"input_ids": padded_input_ids, "attention_mask": attention_masks}
        
        # Non-padded case
        if return_tensors == "pt":
             all_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in all_input_ids]

        return {"input_ids": all_input_ids}

    def decode(self, token_ids: List[int], skip_special_tokens=True) -> str:
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in [self.padding_idx, self.eos_token_id, self.unk_token_id]:
                continue
            tokens.append(self.convert_id_to_token(token_id))
        return " ".join(tokens)
        
    def batch_decode(self, sequences: List[List[int]], skip_special_tokens=True) -> List[str]:
        return [self.decode(seq, skip_special_tokens) for seq in sequences] 