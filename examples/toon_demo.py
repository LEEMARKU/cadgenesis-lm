"""
Demo script for TOON utilities (toon.py).
Run: python toon_demo.py

Shows example serialization, parsing, and token comparison.
"""

from sdk.toon import compare_toon_json_tokens, from_toon, prompt_with_toon, to_toon

sample = [
    {"id": 1, "name": "Widget A", "price": 9.99, "description": "Small widget\n2 colors"},
    {"id": 2, "name": "Widget B", "price": 15.5, "description": "Large | heavy"},
]

if __name__ == "__main__":
    toon_text = to_toon(sample)
    print("--- TOON ---")
    print(toon_text)
    print("\n--- Parsed back ---")
    objs = from_toon(toon_text)
    print(objs)

    cmp = compare_toon_json_tokens(sample)
    print("\n--- Token comparison (approx) ---")
    print(f"JSON tokens approx: {cmp['json_tokens']}")
    print(f"TOON tokens approx: {cmp['toon_tokens']}")
    print(f"Savings: {cmp['savings_tokens']} tokens ({cmp['savings_percent']}%)")

    prompt = "Return ids and names for items with price > 10."
    combined = prompt_with_toon(prompt, sample)
    print("\n--- Prompt with TOON prepended ---")
    print(combined)
