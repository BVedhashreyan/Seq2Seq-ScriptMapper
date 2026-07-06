import torch

def levenshtein_distance(s1, s2):
    """Calculates the minimum edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_metrics(pred_strings, true_strings):
    """
    Computes exact Word Accuracy, Character Accuracy, and CER 
    over collections of decoded native text strings.
    """
    correct_words = 0
    total_words = len(true_strings)
    
    total_char_edits = 0
    total_true_chars = 0
    correct_chars = 0

    for pred_str, true_str in zip(pred_strings, true_strings):
        # 1. Exact Word Accuracy Match
        if pred_str == true_str:
            correct_words += 1

        # 2. Character Error Rate (CER) Metrics
        total_char_edits += levenshtein_distance(pred_str, true_str)
        total_true_chars += len(true_str)

        # 3. Raw Positional Character Matching
        for cp, ct in zip(pred_str, true_str):
            if cp == ct:
                correct_chars += 1

    word_accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0.0
    char_accuracy = (correct_chars / total_true_chars) * 100 if total_true_chars > 0 else 0.0
    cer = total_char_edits / total_true_chars if total_true_chars > 0 else 0.0

    return {
        "word_accuracy": word_accuracy,
        "char_accuracy": char_accuracy,
        "cer": cer
    }