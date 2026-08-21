TOON (Token-Oriented Object Notation)

Overview

TOON is a compact, column-oriented representation for arrays of objects designed to reduce token usage when sending structured data to LLMs. Instead of repeating keys for each object (as with JSON), TOON declares a header of field names once and lists the values underneath — similar to CSV but with explicit escaping for tokens that may confuse LLMs.

Files added

- toon.py — core utilities: serialization, parsing, token estimation, and prompt helper.
- toon_demo.py — small demo showing usage.
- toon_extended.py — enhanced TOON with typed-schema support, chunking, and streaming helpers.
- app_fastapi.py — example FastAPI integration (conversion endpoints, streaming, and optional OpenAI integration).
- requirements.txt — dependencies for running the FastAPI demo.
- toon.ts, toon_node_demo.ts — TypeScript implementation and demo for Node.js/TypeScript projects.
- TOON_INTEGRATION.md — integration notes and usage examples for the added features.

Quick usage

1. Serialize objects to TOON

    from toon import to_toon
    toon_text = to_toon(objects)

2. Parse TOON back to objects

    from toon import from_toon
    objects = from_toon(toon_text)

3. Prepend TOON to a prompt sent to an LLM

    from toon import prompt_with_toon
    combined_prompt = prompt_with_toon(prompt, objects)

Token estimation

- If you have tiktoken installed in your environment, toon.py will use it for accurate token counts for OpenAI-like models.
- If tiktoken is not installed, toon.py falls back to a conservative whitespace-based estimate.

Integration guidance

- Use TOON for large arrays (catalogs, logs, DB exports) to reduce repeated keys and quotes.
- Include a short explanation in the prompt ("Data (TOON): <header>\n<rows> - header fields are: ...") when you first introduce the format to the model to ensure consistent parsing.

Escaping rules

- Backslash (\\) is escaped as \\\\.
- Newline is escaped as \\n inside values.
- Delimiter (default '|') is escaped as \\| inside values.

Notes & next steps

- Consider switching delimiter to '\t' if pipe is common in your data.
- For streaming/very large datasets, consider chunking rows and sending multiple TOON blocks, or compressing externally and passing decompressed chunks to the model.
- If you want strict schema enforcement or typed fields (ints, floats), extend the serializer to include a schema header line with types; parsing can cast values accordingly.

