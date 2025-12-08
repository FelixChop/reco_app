import pytest
from fastapi.testclient import TestClient
from main import app, MODES

client = TestClient(app)

def test_get_modes():
    """Verify that the /modes endpoint returns a list of dictionaries with id and label."""
    response = client.get("/modes")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    for mode in data:
        assert "id" in mode
        assert "label" in mode
        assert mode["id"] in MODES

def test_get_new_session():
    """Verify that /new-session returns a valid UUID string."""
    response = client.get("/new-session")
    assert response.status_code == 200
    assert isinstance(response.json(), str)
    assert len(response.json()) > 0

def test_sample_items_default():
    """Verify that /sample-items returns a list of items."""
    response = client.get("/sample-items?mode=colors&limit=3")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    # Check item structure
    item = data[0]
    assert "id" in item
    assert "name" in item

def test_invalid_mode():
    """Verify strictly that invalid modes return 400."""
    response = client.get("/sample-items?mode=invalid_mode")
    assert response.status_code == 400
