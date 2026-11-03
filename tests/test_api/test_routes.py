"""
Tests for API endpoints using FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self):
        """Health check should always return 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_schema(self):
        """Health response should have required fields."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


class TestCorrectionEndpoint:
    """Tests for the /correct endpoint."""

    def test_correct_requires_text(self):
        """Should return 422 if text field is missing."""
        response = client.post("/api/v1/correct", json={})
        assert response.status_code == 422

    def test_correct_rejects_empty_text(self):
        """Should return 422 for empty text."""
        response = client.post("/api/v1/correct", json={"text": ""})
        assert response.status_code == 422

    def test_correct_validates_num_beams(self):
        """Should reject num_beams outside valid range."""
        response = client.post(
            "/api/v1/correct",
            json={"text": "Hello world.", "num_beams": 100},
        )
        assert response.status_code == 422


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_api_info(self):
        """Root should return API name and version."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
