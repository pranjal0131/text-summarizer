import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)

SAMPLE = (
    "Distributed systems coordinate many machines. "
    "Consensus algorithms keep distributed machines in agreement. "
    "Ice cream is a popular dessert."
)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_summarize_returns_summary_and_metadata():
    response = client.post(
        "/summarize",
        json={"text": SAMPLE, "method": "textrank", "num_sentences": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]
    assert body["input_sentences"] == 3
    assert body["summary_sentences"] == 2
    assert 0 < body["compression_ratio"] <= 1
    assert body["processing_time_ms"] >= 0


def test_num_sentences_clamped_to_input_length():
    response = client.post(
        "/summarize", json={"text": "Just one sentence.", "num_sentences": 10}
    )
    assert response.status_code == 200
    assert response.json()["summary_sentences"] == 1


def test_empty_text_rejected():
    response = client.post("/summarize", json={"text": ""})
    assert response.status_code == 422


def test_invalid_method_rejected():
    response = client.post(
        "/summarize", json={"text": SAMPLE, "method": "quantum"}
    )
    assert response.status_code == 422


def test_num_sentences_bounds_validated():
    response = client.post(
        "/summarize", json={"text": SAMPLE, "num_sentences": 0}
    )
    assert response.status_code == 422
