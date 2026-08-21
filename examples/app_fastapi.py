"""
FastAPI integration demonstrating TOON usage and streaming/chunking support.

Endpoints:
- POST /to_toon -> convert objects to TOON (query params: include_schema, chunk_size)
- POST /from_toon -> parse TOON string back to objects
- POST /stream_toon -> stream TOON chunks as server-sent events (text/event-stream)
- POST /llm_prepare -> build prompt for LLM (returns prompt text and token estimate)
- POST /llm_call -> optional: attempts to call OpenAI if openai package is installed
  and OPENAI_API_KEY is in env

Run locally for testing:
    pip install -r requirements.txt
    uvicorn app_fastapi:app --reload --port 8080

Note: The /llm_call endpoint will not call OpenAI unless openai package is
installed and OPENAI_API_KEY present.
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from sdk.toon_extended import build_prompt_for_llm, chunk_toon, from_toon, stream_toon, to_toon

app = FastAPI(title="TOON FastAPI Integration", version="0.1")


class ObjectsPayload(BaseModel):
    objects: list[dict]
    fields: list[str] | None = None
    types: list[str] | None = None
    include_schema: bool | None = False
    delimiter: str | None = "|"
    chunk_size: int | None = None


class ToonStringPayload(BaseModel):
    toon: str
    delimiter: str | None = "|"


class LLMRequest(BaseModel):
    instruction: str
    objects: list[dict]
    fields: list[str] | None = None
    types: list[str] | None = None
    include_schema: bool | None = False
    delimiter: str | None = "|"
    chunk_size: int | None = None
    # Optional model and other LLM params
    model: str | None = None
    # If streaming true, return generator for chunks instead of calling remote LLM
    streaming: bool | None = False


@app.post("/to_toon")
async def endpoint_to_toon(payload: ObjectsPayload):
    toon_text = to_toon(
        payload.objects,
        fields=payload.fields,
        delimiter=payload.delimiter or "|",
        types=payload.types,
        include_schema=payload.include_schema,
    )
    return PlainTextResponse(toon_text)


@app.post("/from_toon")
async def endpoint_from_toon(payload: ToonStringPayload):
    try:
        objs = from_toon(payload.toon, delimiter=payload.delimiter or "|")
        return JSONResponse(objs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/stream_toon")
async def endpoint_stream_toon(payload: ObjectsPayload):
    chunk_size = payload.chunk_size or 100
    gen = stream_toon(
        payload.objects,
        chunk_size=chunk_size,
        fields=payload.fields,
        types=payload.types,
        include_schema=payload.include_schema,
        delimiter=payload.delimiter or "|",
    )

    def event_generator():
        for c in gen:
            # yield a simple JSON per event
            yield f"data: {json.dumps(c, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/llm_prepare")
async def endpoint_llm_prepare(req: LLMRequest):
    prepared = build_prompt_for_llm(
        req.instruction,
        req.objects,
        fields=req.fields,
        types=req.types,
        include_schema=req.include_schema,
        delimiter=req.delimiter or "|",
    )
    return prepared


@app.post("/llm_call")
async def endpoint_llm_call(req: LLMRequest):
    # If streaming flag is set, return chunks (useful for user wanting
    # chunk-by-chunk local handling)
    if req.streaming:
        chunks = chunk_toon(
            req.objects,
            chunk_size=req.chunk_size or 100,
            fields=req.fields,
            types=req.types,
            include_schema=req.include_schema,
            delimiter=req.delimiter or "|",
        )
        return JSONResponse({"chunks": chunks})

    # Otherwise build prompt and optionally call OpenAI if configured
    prepared = build_prompt_for_llm(
        req.instruction,
        req.objects,
        fields=req.fields,
        types=req.types,
        include_schema=req.include_schema,
        delimiter=req.delimiter or "|",
    )
    prompt_text = prepared["prompt_text"]
    model = req.model or os.environ.get("TOON_DEFAULT_MODEL", "gpt-4")

    # Try to call OpenAI if available
    try:
        import openai

        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            return JSONResponse(
                {"error": "OPENAI_API_KEY not set; returning prepared prompt", "prepared": prepared}
            )
        openai.api_key = openai_api_key
        resp = openai.ChatCompletion.create(
            model=model, messages=[{"role": "user", "content": prompt_text}], temperature=0
        )
        return JSONResponse({"openai_response": resp, "prepared": prepared})
    except Exception as e:
        # openai not installed or call failed
        return JSONResponse(
            {"error": "OpenAI call not performed", "reason": str(e), "prepared": prepared}
        )


# local imports

if __name__ == "__main__":
    print("Run with: uvicorn app_fastapi:app --reload --port 8080")
