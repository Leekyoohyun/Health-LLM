#!/usr/bin/env python3
"""
Tokenizer max_length 문제 수정
inferer.py에서 tokenizer.model_max_length를 명시적으로 설정
"""

import torch
from transformers import AutoTokenizer

# Check current tokenizer settings
print("=" * 80)
print("Checking Tokenizer Settings")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b")

print(f"\nDefault tokenizer settings:")
print(f"  model_max_length: {tokenizer.model_max_length}")
print(f"  max_len_single_sentence: {tokenizer.max_len_single_sentence if hasattr(tokenizer, 'max_len_single_sentence') else 'N/A'}")
print(f"  max_len_sentences_pair: {tokenizer.max_len_sentences_pair if hasattr(tokenizer, 'max_len_sentences_pair') else 'N/A'}")

# Test input
test_input = "The recent 14-days sensor readings show: " + "[Steps]: " + str([1000.0] * 100)
tokens = tokenizer(test_input, return_tensors="pt", truncation=True)
print(f"\nTest input tokens: {tokens['input_ids'].shape[1]}")

# Fix: set model_max_length to 2048
print("\n" + "=" * 80)
print("Setting model_max_length to 2048")
print("=" * 80)

tokenizer.model_max_length = 2048
tokens_fixed = tokenizer(test_input, return_tensors="pt", truncation=True, max_length=2048)
print(f"After fix, test input tokens: {tokens_fixed['input_ids'].shape[1]}")

print("\n✅ To fix in inferer.py, add after line 62:")
print("   self.data_handler.tokenizer.model_max_length = model_max_length")
