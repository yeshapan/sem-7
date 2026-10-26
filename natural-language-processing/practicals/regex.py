# -*- coding: utf-8 -*-
"""
Module: Regex Operations
Validates Indian phone numbers, performs case-insensitive replacement, and filters vowel words.
"""

import re


def validate_indian_phones(numbers: list) -> list:
    """Validates numbers matching +91-9XXXXXXXXX or +91-8XXXXXXXXX."""
    pattern = r'^\+91-[98]\d{9}$'
    return [(num, bool(re.fullmatch(pattern, num.strip()))) for num in numbers]


def regex_replace(text: str, target: str, replacement: str) -> str:
    """Case-insensitive pattern replace."""
    return re.sub(target, replacement, text, flags=re.IGNORECASE)


def find_vowel_starting_words(text: str) -> list:
    """Finds all words beginning with a vowel."""
    return re.findall(r'\b[AEIOUaeiou]\w*', text)


if __name__ == "__main__":
    # 1. Phone validation
    test_numbers = ["+91-9876543210", "+91-8123456789", "+91-7123456789", "9876543210"]
    print("--- Phone Validation ---")
    for num, is_valid in validate_indian_phones(test_numbers):
        print(f"{num:<16} : {'Valid' if is_valid else 'Invalid'}")

    # 2. Search & replace
    raw_text = "Machine learning engineers use machine algorithms to build modern machine models."
    replaced_text = regex_replace(raw_text, "machine", "AI")
    print("\n--- Search and Replace ---")
    print(replaced_text)

    # 3. Words starting with vowels
    passage = "An intelligent agent operates under uncertainty with optimal accuracy."
    print("\n--- Words Starting with Vowels ---")
    print(find_vowel_starting_words(passage))