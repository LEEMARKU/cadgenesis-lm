TOON Integration & Examples

This file documents the added integration features, typed-schema support, chunking/streaming, and Node.js/TypeScript implementation.

Python files added:
- toon_extended.py — enhanced TOON with typed-schema and chunking/streaming helpers.
- app_fastapi.py — FastAPI endpoints demonstrating conversion, streaming, and optional OpenAI calling.
- requirements.txt — Python dependencies for running the FastAPI server.

Node/TypeScript files added:
- toon.ts — TypeScript implementation of TOON with typed-schema and chunking.
- toon_node_demo.ts — demo usage for the TypeScript module.

How to run the FastAPI demo (Python)
1. (Optional) Create a virtualenv and activate it.
2. Install dependencies:
    pip install -r requirements.txt
3. Run the server:
    uvicorn app_fastapi:app --reload --port 8080
4. Use curl or Postman to hit endpoints:
    POST http://localhost:8080/to_toon with JSON body {"objects": [...]}
    POST http://localhost:8080/llm_prepare with JSON body {"instruction":"...","objects":[...]} to get prompt and token estimate.

Streaming chunks
- POST to /stream_toon with JSON {"objects": [...], "chunk_size": 50}
- The endpoint responds with Server-Sent Events (text/event-stream) containing JSON objects for each chunk.

Typed schema
- Set include_schema=true when generating TOON; types may be provided, otherwise they are inferred from the first object.
- Example of a TOON with schema:
    id|name|price
    int|str|float
    1|Widget A|9.99
    2|Widget B|15.5
- When parsing, the schema line is detected and casting is applied (int, float, bool, str)

Node/TypeScript
- Use the TypeScript module by importing the functions from toon.ts. See toon_node_demo.ts for example code.

Next steps
- If desired, integrate the /llm_call endpoint to actually call your model provider: set OPENAI_API_KEY in environment and ensure openai package is installed.
- Add unit tests for the serializer/parser and type casting (recommended before production use).
