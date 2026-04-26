"""End-to-end FastAPI tests using TestClient."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_catalog_containers(client):
    r = client.get("/api/loading/catalog/containers")
    assert r.status_code == 200
    body = r.json()
    codes = {c["code"] for c in body}
    assert "40HC" in codes


def test_catalog_cargo_presets(client):
    r = client.get("/api/loading/catalog/cargo-presets")
    assert r.status_code == 200
    body = r.json()
    assert any(p["code"] == "eur_pallet_heavy" for p in body)


def test_catalog_imdg(client):
    r = client.get("/api/loading/catalog/imdg-segregation")
    assert r.status_code == 200
    body = r.json()
    assert len(body["classes"]) == 10
    assert len(body["matrix"]) == 10


def test_solve_with_extreme_points(client, eur_pallets_10):
    payload = {
        "container_code": "40HC",
        "items": [it.model_dump(mode="json") for it in eur_pallets_10],
        "algorithm": "extreme_points",
        "seed": 1,
    }
    r = client.post("/api/loading/solve", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["algorithm"] == "extreme_points"
    assert len(body["placements"]) == 10
    assert body["unplaced_item_ids"] == []


def test_solve_compare(client, mixed_bag):
    payload = {
        "container_code": "40HC",
        "items": [it.model_dump(mode="json") for it in mixed_bag],
        "algorithm_a": "bl",
        "algorithm_b": "extreme_points",
    }
    r = client.post("/api/loading/compare", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "a" in body and "b" in body
    assert body["a"]["algorithm"] == "bl"
    assert body["b"]["algorithm"] == "extreme_points"


def test_ws_solve_stream(client, eur_pallets_10):
    payload = {
        "container_code": "40HC",
        "items": [it.model_dump(mode="json") for it in eur_pallets_10],
        "algorithm": "bl",
    }
    with client.websocket_connect("/api/loading/solve/stream") as ws:
        ws.send_text(__import__("json").dumps(payload))
        n_steps = 0
        seen_done = False
        # Read until we see {"done": True}
        for _ in range(40):
            msg = ws.receive_json()
            if msg.get("done"):
                seen_done = True
                break
            n_steps += 1
        assert seen_done
        assert n_steps == 10
