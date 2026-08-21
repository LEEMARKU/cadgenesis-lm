"""cadgenesis.tokenizer.cad_tokens
================================
Aggregate CAD token registry for the CADGenesis-LM v6.0 vocabulary.

Combines every token family definition table with the special control tokens
into one place, so callers can enumerate the full CAD token universe without
importing each family module individually.
"""

from cadgenesis.tokenizer.assembly_tokens import ALL_ASSEMBLY_TOKENS
from cadgenesis.tokenizer.constraint_tokens import ALL_CONSTRAINT_TOKENS
from cadgenesis.tokenizer.feature_tokens import ALL_FEATURE_TOKENS
from cadgenesis.tokenizer.geometry_tokens import ALL_GEOMETRY_TOKENS
from cadgenesis.tokenizer.manufacturing_tokens import ALL_MANUFACTURING_TOKENS
from cadgenesis.tokenizer.material_tokens import ALL_MATERIAL_TOKENS
from cadgenesis.tokenizer.numeric_tokens import (
    all_numeric_token_strings,
    build_numeric_token_table,
)
from cadgenesis.tokenizer.simulation_tokens import ALL_SIMULATION_TOKENS
from cadgenesis.tokenizer.vocabulary import (
    AGENT_TOKEN,
    ANSWER_TOKEN,
    ASSEMBLY_END_TOKEN,
    ASSEMBLY_START_TOKEN,
    BOS_TOKEN,
    CAD_END_TOKEN,
    CAD_START_TOKEN,
    CLS_TOKEN,
    CONSTRAINT_END_TOKEN,
    CONSTRAINT_START_TOKEN,
    EOS_TOKEN,
    MANUF_END_TOKEN,
    MANUF_START_TOKEN,
    MASK_TOKEN,
    MATERIAL_END_TOKEN,
    MATERIAL_START_TOKEN,
    MEMORY_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    SIM_END_TOKEN,
    SIM_START_TOKEN,
    THINK_TOKEN,
    UNK_TOKEN,
)

SPECIAL_CAD_TOKENS: tuple[str, ...] = (
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    MASK_TOKEN,
    CLS_TOKEN,
    SEP_TOKEN,
    CAD_START_TOKEN,
    CAD_END_TOKEN,
    CONSTRAINT_START_TOKEN,
    CONSTRAINT_END_TOKEN,
    ASSEMBLY_START_TOKEN,
    ASSEMBLY_END_TOKEN,
    MATERIAL_START_TOKEN,
    MATERIAL_END_TOKEN,
    MANUF_START_TOKEN,
    MANUF_END_TOKEN,
    SIM_START_TOKEN,
    SIM_END_TOKEN,
    THINK_TOKEN,
    ANSWER_TOKEN,
    MEMORY_TOKEN,
    AGENT_TOKEN,
)

FAMILY_TOKEN_TABLES: dict[str, list[tuple[str, str]]] = {
    "geometry": ALL_GEOMETRY_TOKENS,
    "feature": ALL_FEATURE_TOKENS,
    "constraint": ALL_CONSTRAINT_TOKENS,
    "material": ALL_MATERIAL_TOKENS,
    "assembly": ALL_ASSEMBLY_TOKENS,
    "manufacturing": ALL_MANUFACTURING_TOKENS,
    "simulation": ALL_SIMULATION_TOKENS,
}

STATIC_CAD_TOKEN_TABLES: list[tuple[str, str]] = (
    ALL_GEOMETRY_TOKENS
    + ALL_FEATURE_TOKENS
    + ALL_CONSTRAINT_TOKENS
    + ALL_MATERIAL_TOKENS
    + ALL_ASSEMBLY_TOKENS
    + ALL_MANUFACTURING_TOKENS
    + ALL_SIMULATION_TOKENS
)


def all_cad_token_tables() -> list[tuple[str, str]]:
    """Return every statically-defined CAD token plus generated numeric bins."""
    return list(STATIC_CAD_TOKEN_TABLES) + build_numeric_token_table()


def all_cad_token_strings() -> list[str]:
    """Return every CAD (non-special) token string in the registry."""
    return [token for token, _ in all_cad_token_tables()]


__all__ = [
    "FAMILY_TOKEN_TABLES",
    "SPECIAL_CAD_TOKENS",
    "STATIC_CAD_TOKEN_TABLES",
    "all_cad_token_strings",
    "all_cad_token_tables",
    "all_numeric_token_strings",
]
