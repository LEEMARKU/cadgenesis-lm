"""cadgenesis.tokenizer.numeric_tokens
====================================
Numeric token definitions.

Numeric tokens (``NUM_`` / ``ANG_`` / ``RATIO_``) are quantizer bins, so the
token table is generated from the numeric quantizers rather than hard-coded.
"""

from cadgenesis.tokenizer.numeric import NumericTokenizer


def build_numeric_token_table() -> list[tuple[str, str]]:
    """Generate ``(token_string, description)`` for every numeric quantizer bin."""
    table: list[tuple[str, str]] = []
    for quantizer, suffix, unit in (
        (NumericTokenizer.length_quantizer(), "mm", "Length"),
        (NumericTokenizer.angle_quantizer(), "deg", "Angle"),
        (NumericTokenizer.ratio_quantizer(), "", "Ratio"),
    ):
        for token in quantizer.all_token_strings():
            value = quantizer.decode(token)
            if unit == "Angle":
                desc = f"Angle bin: {value:.2f} deg"
            else:
                desc = f"{unit} bin: {value:.4f} {suffix}"
            table.append((token, desc))
    return table


def all_numeric_token_strings() -> list[str]:
    """Return every numeric token string (quantizer bins)."""
    return [token for token, _ in build_numeric_token_table()]


__all__ = [
    "NumericTokenizer",
    "all_numeric_token_strings",
    "build_numeric_token_table",
]
