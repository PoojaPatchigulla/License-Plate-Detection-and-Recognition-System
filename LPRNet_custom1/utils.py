# 4. utils.py (for decoding CTC predictions)

import torch

def ctc_decode(logits, alphabet):
    """
    Performs greedy CTC decoding on logits.

    Args:
        logits: Tensor of shape [T, B, C] (output of LPRNet)
        alphabet: List of characters (excluding blank)

    Returns:
        List of decoded strings, one per batch item.
    """
    logits = logits.permute(1, 0, 2)  # [B, T, C]
    preds = torch.argmax(logits, dim=2)  # [B, T]

    decoded_strings = []
    blank_idx = len(alphabet)  # CTC expects blank to be the last index

    for pred in preds:
        prev = -1
        string = ''
        for p in pred:
            p = p.item()
            if p != prev and p != blank_idx:
                if p < len(alphabet):
                    string += alphabet[p]
            prev = p
        decoded_strings.append(string)

    return decoded_strings
