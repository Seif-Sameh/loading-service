"""POST /solve and WS /solve/stream — the main optimisation endpoints."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.algorithms.base import SolveResult, StepEvent
from app.catalog.loader import get_container
from app.schemas import (
    CargoItem,
    Container,
    SolveRequest,
)
from app.services.solver import SolverService

router = APIRouter(prefix="/api/loading", tags=["solve"])


def _resolve(req: SolveRequest) -> tuple[Container, list[CargoItem]]:
    cont = get_container(req.container_code.value)
    return cont, list(req.items)


@router.post("/solve")
def post_solve(req: SolveRequest) -> dict[str, Any]:
    """Run the algorithm to completion and return the full :class:`SolveResult`."""
    cont, items = _resolve(req)
    result = SolverService.run(
        container=cont, items=items, algorithm=req.algorithm, seed=req.seed
    )
    return result.model_dump(mode="json")


@router.post("/compare")
def post_compare(payload: dict[str, Any]) -> dict[str, Any]:
    """Run two algorithms on the same voyage and return both :class:`SolveResult` payloads.

    Body schema:
        {
            "container_code": "40HC",
            "items": [...],
            "algorithm_a": "extreme_points",
            "algorithm_b": "ppo",
            "seed": 42
        }
    """
    a_req = SolveRequest(
        container_code=payload["container_code"],
        items=payload["items"],
        algorithm=payload.get("algorithm_a", "extreme_points"),
        seed=payload.get("seed"),
    )
    b_req = SolveRequest(
        container_code=payload["container_code"],
        items=payload["items"],
        algorithm=payload.get("algorithm_b", "bl"),
        seed=payload.get("seed"),
    )
    cont_a, items_a = _resolve(a_req)
    cont_b, items_b = _resolve(b_req)
    a = SolverService.run(container=cont_a, items=items_a, algorithm=a_req.algorithm, seed=a_req.seed)
    b = SolverService.run(container=cont_b, items=items_b, algorithm=b_req.algorithm, seed=b_req.seed)
    return {
        "a": a.model_dump(mode="json"),
        "b": b.model_dump(mode="json"),
    }


@router.websocket("/solve/stream")
async def ws_solve_stream(websocket: WebSocket) -> None:
    """Stream placements one-at-a-time so the 3D scene can animate them.

    Protocol:
    1. Client opens the socket.
    2. Client sends a JSON :class:`SolveRequest`.
    3. Server emits one frame per placement, then a final ``{"done": true, "summary": ...}``.
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            req = SolveRequest.model_validate(payload)
            cont, items = _resolve(req)
            async for event in SolverService.stream(
                container=cont, items=items, algorithm=req.algorithm, seed=req.seed
            ):
                if isinstance(event, StepEvent):
                    await websocket.send_json(_step_event_to_dict(event))
                elif isinstance(event, SolveResult):
                    await websocket.send_json(
                        {"done": True, "summary": event.model_dump(mode="json")}
                    )
            await websocket.send_json({"done": True})
    except WebSocketDisconnect:
        return


def _step_event_to_dict(ev: StepEvent) -> dict[str, Any]:
    return {
        "step": ev.step,
        "remaining": ev.remaining,
        "placement": ev.placement.model_dump(mode="json"),
    }
