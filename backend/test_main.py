import os
os.environ["SENSEAI_DISABLE_TRANSFORMER"] = "1"

import pytest
from fastapi.testclient import TestClient
from . import main
from uuid import uuid4

client = TestClient(main.app)

@pytest.fixture(autouse=True)
def local_engine(monkeypatch):
    monkeypatch.setattr(main, "_encoder_model", lambda: None)
    main.app.dependency_overrides[main.get_current_user] = lambda: main.User(id=91001, email="tester@example.com")
    yield
    main.app.dependency_overrides.pop(main.get_current_user, None)

def test_bank_financial_and_river_contexts():
    if main._wordnet() is None: pytest.skip("Install NLTK WordNet data to run integration tests")
    financial = client.post("/api/wsd/analyze", json={"text":"She deposited her paycheck at the bank before work.","target":"bank"}).json()
    river = client.post("/api/wsd/analyze", json={"text":"The fisherman sat on the bank beside the flowing river.","target":"bank"}).json()
    assert financial["status"] == river["status"] == "ok"
    assert financial["prediction"]["synset"] != river["prediction"]["synset"]

def test_bat_animal_and_sport_contexts_keep_real_wordnet_candidates():
    if main._wordnet() is None: pytest.skip("Install NLTK WordNet data to run integration tests")
    animal = client.post("/api/wsd/analyze", json={"text":"A bat flew out of the cave at dusk.","target":"bat"}).json()
    sport = client.post("/api/wsd/analyze", json={"text":"She swung the bat and hit the ball over the fence.","target":"bat"}).json()
    assert animal["status"] == sport["status"] == "ok"
    assert "bat.n.01" in {s["synset"] for s in animal["candidates"]}
    assert {"bat.n.02", "bat.n.05"}.intersection(s["synset"] for s in sport["candidates"])

def test_repeated_target_occurrence_is_highlighted():
    if main._wordnet() is None: pytest.skip("Install NLTK WordNet data to run integration tests")
    response=client.post("/api/wsd/analyze",json={"text":"The bank was near another bank.","target":"bank","occurrence":2})
    assert response.status_code==200
    assert response.json()["occurrence"]==2
    assert response.json()["sentence"][response.json()["start"]:response.json()["end"]]=="bank"

def test_unknown_word_returns_honest_unknown_result():
    if main._wordnet() is None: pytest.skip("Install NLTK WordNet data to run integration tests")
    response=client.post("/api/wsd/analyze",json={"text":"Qzxv is a made up token.","target":"Qzxv"})
    assert response.status_code==200
    assert response.json()["status"]=="unknown"

def test_invalid_request_has_validation_error():
    response=client.post("/api/wsd/analyze",json={"text":"","target":"bank"})
    assert response.status_code==422

def test_missing_target_is_clear_validation_message():
    response=client.post("/api/wsd/analyze",json={"text":"The river ran fast.","target":"bank"})
    assert response.status_code==422
    assert "not found" in response.json()["detail"]

def test_evaluation_metrics_use_submitted_gold_labels(monkeypatch):
    predictions=iter(["bank.n.01","bank.n.02"])
    monkeypatch.setattr(main,"analyze",lambda item,user_id=None:{"prediction":{"synset":next(predictions)}})
    response=client.post("/api/evaluation/run",json={"dataset":"Local labels","items":[{"text":"Text one","target":"bank"},{"text":"Text two","target":"bank"}],"labels":["bank.n.01","bank.n.01"]})
    assert response.status_code==200
    data=response.json()
    assert data["sample_count"]==2
    assert data["accuracy"]==0.5
    assert data["confusion_matrix"]["values"]["bank.n.01"]["bank.n.02"]==1

def test_local_account_register_login_and_logout():
    main.app.dependency_overrides.pop(main.get_current_user, None)
    email=f"{uuid4().hex}@example.com"
    try:
        created=client.post("/api/auth/register",json={"email":email,"password":"a-strong-local-password"})
        assert created.status_code==200
        token=created.json()["access_token"]
        assert client.get("/api/auth/me",headers={"Authorization":f"Bearer {token}"}).json()["email"]==email
        wrong=client.post("/api/auth/login",json={"email":email,"password":"incorrect-password"})
        assert wrong.status_code==401
        signed_in=client.post("/api/auth/login",json={"email":email,"password":"a-strong-local-password"})
        assert signed_in.status_code==200
        token=signed_in.json()["access_token"]
        assert client.post("/api/auth/logout",headers={"Authorization":f"Bearer {token}"}).status_code==200
        assert client.get("/api/auth/me",headers={"Authorization":f"Bearer {token}"}).status_code==401
    finally:
        main.app.dependency_overrides[main.get_current_user]=lambda: main.User(id=91001,email="tester@example.com")

def test_analysis_requires_sign_in():
    main.app.dependency_overrides.pop(main.get_current_user, None)
    try:
        response=client.post("/api/wsd/analyze",json={"text":"The bank is open.","target":"bank"})
        assert response.status_code==401
    finally:
        main.app.dependency_overrides[main.get_current_user]=lambda: main.User(id=91001,email="tester@example.com")
