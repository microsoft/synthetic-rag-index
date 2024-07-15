import hashlib
from typing import Optional

import nltk

from helpers.logging import logger


def detect_extension(file_path: str) -> str:
    """
    Detect the extension of a file path.

    For example, if the file path is "file.txt", the extension will be ".txt".

    Return the extension in lowercase.
    """
    return "." + file_path.lower().split(".")[-1]


def replace_root_path(file_path: str, new_root: str) -> str:
    """
    Replace the root path of a file path.

    For example, if the file path is "raw/2022-01-01/file.txt" and the new root is "fact", the new file path will be "fact/2022-01-01/file.txt".
    """
    return new_root + "/" + "".join(file_path.split("/")[1:])


def replace_extension(file_path: str, new_extension: str) -> str:
    """
    Replace the extension of a file path.

    For example, if the file path is "file.txt" and the new extension is "json", the new file path will be "file.json".
    """
    return "".join(file_path.split(".")[:-1]) + new_extension


def hash_text(text: str) -> str:
    """
    Hash a text using MD5.

    MD5 is as of today the fastest hash function available.

    Return the MD5 hash of the text.
    """
    # deepcode ignore InsecureHash: It has collision vulnerabilities, but it is not a concern in this context. It is used for deduplication not for PII data.
    return hashlib.md5(string=text.encode(), usedforsecurity=False).hexdigest()


def sanitize_text(
    text: str,
    max_word_length: int = 1000,
    min_words_per_line: int = 5,
) -> Optional[str]:
    """
    Cleans text, return nothing if it should be skipped.

    Cleaning removes lines with no end marks or with too few words. After line filtering, pages are filtered out if they have too few sentences based on a simple count of end marks.

    This functions implement "Clean Crawled Corpus" from Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (https://arxiv.org/abs/1910.10683).

    Return the cleaned text.
    """
    markdown_start_marks = ("#", "*", "-", "+", "_", ">", "|", "`")
    text_end_ellipsis = "..."
    text_end_marks = (".", "?", "!", '"')
    policy_substrings = [
        "cookie policy",
        "privacy policy",
        "terms of use",
        "use cookies",
        "use of cookies",
        "uses cookies",
    ]

    lines = text.splitlines()  # Split by lines
    valid_lines = []

    def _line_has_too_long_word(line):
        """
        Check if a line contains a word that is too long.
        """
        for word in line.split():  # Split by whitespace
            if len(word) > max_word_length:  # Check if word is too long
                return True
        return False

    for line in lines:
        line = line.strip()
        if _line_has_too_long_word(line):  # Skip lines with too long words
            continue
        if not line.endswith(text_end_marks) or line.endswith(
            text_end_ellipsis
        ):  # Skip lines without end marks
            if not line.startswith(markdown_start_marks):  # Keep markdown lines
                continue
        if len(line.split()) < min_words_per_line:  # Skip lines with too few words
            continue
        line_lower = line.lower()
        if "lorem ipsum" in line_lower:  # Skip entire page if it contains lorem ipsum
            logger.info("Lorem ipsum detected, skipping page")
            return
        if any(p in line_lower for p in policy_substrings):  # Skip policy lines
            continue
        valid_lines.append(line)

    if not valid_lines:  # Skip empty pages
        logger.info("Empty page, skipping")
        return

    logger.info(f"Page cleaned, {len(lines)/len(valid_lines):.2f}x reduction")
    return "\n".join(valid_lines).strip()


def has_excessive_repetition(
    text: str,
    threshold_ratio: float = 1.0,
) -> bool:
    """
    Check if there is repeated content in the input text. Excessive repetition is often linked with uninformative content and can be used to determine whether it is low-quality text.

    Threshold ratio is relative to the recommended values in the paper. The default value is 1.0, which corresponds to the recommended values.

    This function implements "Repetition Removal" from Scaling Language Models: Methods, Analysis & Insights from Training Gopher (https://arxiv.org/abs/2112.11446).

    Return True if the text is considered to have excessive repetition, False otherwise.
    """
    duplicate_line_character_faction = (
        0.2 * threshold_ratio
    )  # Duplicate line character fraction
    duplicate_line_fraction = 0.3 * threshold_ratio  # Duplicate line fraction

    dup_line = 0
    dup_line_chars = 0
    line_count = 0
    visit_lines = {}

    # Check for repeated lines
    for line in text.splitlines():
        line_hash = hash_text(line)
        if line_hash in visit_lines:
            dup_line += 1
            dup_line_chars += len(line)
        visit_lines[line_hash] = True
        line_count += 1

    if (
        float(dup_line) / line_count > duplicate_line_fraction
    ):  # Excessive repeated lines
        return True

    if (
        float(dup_line_chars) / len(text) > duplicate_line_character_faction
    ):  # Excessive repeated characters
        return True

    top_ngram_character_fractions = [
        (2, 0.2 * threshold_ratio),  # Top 2-gram character fraction
        (3, 0.18 * threshold_ratio),  # Top 3-gram character fraction
        (4, 0.16 * threshold_ratio),  # Top 4-gram character fraction
    ]
    for ngram, threshold in top_ngram_character_fractions:
        bgs = nltk.ngrams(text.split(), ngram)
        fdist = nltk.FreqDist(bgs)
        for word_list, repeat in fdist.items():
            char_count = sum([len(word) for word in word_list])
            if char_count * (repeat - 1) / len(text) > threshold:
                return True

    duplicate_ngram_character_fractions = [
        (5, 0.15 * threshold_ratio),  # Duplicate 5-gram character fraction
        (6, 0.14 * threshold_ratio),  # Duplicate 6-gram character fraction
        (7, 0.13 * threshold_ratio),  # Duplicate 7-gram character fraction
        (8, 0.12 * threshold_ratio),  # Duplicate 8-gram character fraction
        (9, 0.11 * threshold_ratio),  # Duplicate 9-gram character fraction
        (10, 0.10 * threshold_ratio),  # Duplicate 10-gram character fraction
    ]
    for ngram, threshold in duplicate_ngram_character_fractions:
        fdist = {}
        word_list = text.split()
        mark = [0] * len(word_list)
        for i in range(len(word_list) - ngram + 1):
            bag = tuple(word_list[i : i + ngram])
            if bag in fdist:
                for j in range(i, i + ngram):
                    mark[j] = len(word_list[j])
                fdist[bag] += 1
            else:
                fdist[bag] = 1

        if sum(mark) / float(len(text)) > threshold:
            return True

    return False
