"""
cadgenesis.serving.websocket
============================
WebSocket support for the CADGenesis serving stack.

- Real-time inference: client sends a prompt, server streams token events
- Progress updates: ``type: progress`` events for long jobs
- Event streaming: arbitrary platform events (``type: event``)

Protocol (JSON messages)::

    client -> {"type": "generate", "text": "...", "max_len": 64}
    server -> {"type": "token", "index": 0, "token": "BOX"}
    server -> {"type": "done", "text": "...", "confidence": 0.9}
    server -> {"type": "error", "message": "..."}
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger("cadgenesis.serving.websocket")


async def websocket_endpoint(websocket: Any, state: Any) -> None:
    """Handle one WebSocket connection (FastAPI ``WebSocket`` + serve state)."""
    await websocket.accept()
    await websocket.send_json({"type": "hello", "version": "v1", "ts": time.time()})
    try:
        while True:
            message = await websocket.receive_json()
            kind = message.get("type")
            if kind == "ping":
                await websocket.send_json({"type": "pong"})
            elif kind == "generate":
                await _handle_generate(websocket, state, message)
            elif kind == "subscribe":
                await websocket.send_json(
                    {
                        "type": "event",
                        "channel": message.get("channel", "platform"),
                        "message": "subscribed",
                    }
                )
            else:
                await websocket.send_json(
                    {"type": "error", "message": f"unknown message type {kind!r}"}
                )
    except Exception as exc:
        logger.debug("websocket session ended: %s", exc)


async def _handle_generate(websocket: Any, state: Any, message: dict[str, Any]) -> None:
    from starlette.concurrency import run_in_threadpool

    text = str(message.get("text", ""))
    if not text:
        await websocket.send_json({"type": "error", "message": "empty prompt"})
        return
    if state.lifecycle is None or "default" not in state.lifecycle:
        await websocket.send_json({"type": "error", "message": "no model loaded"})
        return
    max_len = int(message.get("max_len", 64))
    await websocket.send_json({"type": "progress", "stage": "encoding", "pct": 10})
    try:
        engine = state.lifecycle.engine("default")
        result = await run_in_threadpool(engine.greedy, text, max_len=max_len)
        for index, token in enumerate(result.tokens):
            await websocket.send_json({"type": "token", "index": index, "token": token})
            await websocket.send_json(
                {
                    "type": "progress",
                    "stage": "decoding",
                    "pct": min(90, 10 + int(80 * (index + 1) / max(1, len(result.tokens)))),
                }
            )
        await websocket.send_json(
            {
                "type": "done",
                "text": " ".join(result.tokens),
                "confidence": round(float(result.confidence), 6),
                "toon": result.toon,
            }
        )
    except Exception as exc:
        logger.exception("websocket generation failed")
        await websocket.send_json({"type": "error", "message": str(exc)})


def encode_event(channel: str, payload: dict[str, Any]) -> str:
    """Encode an outbound event-stream message for non-FastAPI transports."""
    return json.dumps({"type": "event", "channel": channel, **payload})


__all__ = ["encode_event", "websocket_endpoint"]
